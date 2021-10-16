import inspect
import pkgutil

from typing import Any, Dict, List, Set, Tuple

import sphinx

from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autosummary import get_documenter
from sphinx.ext.autosummary.generate import AutosummaryRenderer
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import logging, split_full_qualified_name
from sphinx.util.inspect import getall, safe_getattr

logger = logging.getLogger(__name__)


class ModuleScanner:
    def __init__(self, app: Any, obj: Any) -> None:
        self.app = app
        self.object = obj

    def get_object_type(self, name: str, value: Any) -> str:
        return get_documenter(self.app, value, self.object).objtype

    def is_skipped(self, name: str, value: Any, objtype: str) -> bool:
        try:
            return self.app.emit_firstresult("autodoc-skip-member", objtype, name, value, False, {})
        except Exception as exc:
            logger.warning(
                __(
                    "autosummary: failed to determine %r to be documented, "
                    "the following exception was raised:\n%s"
                ),
                name,
                exc,
                type="autosummary",
            )
            return False

    def scan(self, imported_members: bool) -> List[str]:
        members = []
        for name in getall(self.object) or dir(self.object):
            try:
                value = safe_getattr(self.object, name)
            except AttributeError:
                value = None

            objtype = self.get_object_type(name, value)
            if self.is_skipped(name, value, objtype):
                continue

            try:
                if inspect.ismodule(value):
                    imported = True
                elif safe_getattr(value, "__module__") != self.object.__name__:
                    imported = True
                else:
                    imported = False
            except AttributeError:
                imported = False

            if imported_members:
                # list all members up
                members.append(name)
            elif imported is False:
                # list not-imported members up
                members.append(name)

        return members


def generate_autosummary_content(
    name: str,
    obj: Any,
    parent: Any,
    template: AutosummaryRenderer,
    template_name: str,
    imported_members: bool,
    app: Any,
    recursive: bool,
    context: Dict,
    modname: str = None,
    qualname: str = None,
) -> str:

    doc = get_documenter(app, obj, parent)

    def skip_member(obj: Any, name: str, objtype: str) -> bool:
        try:
            return app.emit_firstresult("autodoc-skip-member", objtype, name, obj, False, {})
        except Exception as exc:
            logger.warning(
                __(
                    "autosummary: failed to determine %r to be documented, "
                    "the following exception was raised:\n%s"
                ),
                name,
                exc,
                type="autosummary",
            )
            return False

    def get_class_members(obj: Any) -> Dict[str, Any]:
        members = sphinx.ext.autodoc.get_class_members(obj, [qualname], safe_getattr)
        return {name: member.object for name, member in members.items()}

    def get_module_members(obj: Any) -> Dict[str, Any]:
        members = {}
        for name in getall(obj) or dir(obj):
            try:
                members[name] = safe_getattr(obj, name)
            except AttributeError:
                continue
        return members

    def get_all_members(obj: Any) -> Dict[str, Any]:
        if doc.objtype == "module":
            return get_module_members(obj)
        elif doc.objtype == "class":
            return get_class_members(obj)
        return {}

    def get_members(
        obj: Any, types: Set[str], include_public: List[str] = [], imported: bool = True
    ) -> Tuple[List[str], List[str]]:
        items: List[str] = []
        public: List[str] = []

        all_members = get_all_members(obj)
        for name, value in all_members.items():
            documenter = get_documenter(app, value, obj)
            if documenter.objtype in types:
                # skip imported members if expected
                if imported or getattr(value, "__module__", None) == obj.__name__:
                    skipped = skip_member(value, name, documenter.objtype)
                    if skipped is True:
                        pass
                    elif skipped is False:
                        # show the member forcedly
                        items.append(name)
                        public.append(name)
                    else:
                        items.append(name)
                        if name in include_public or not name.startswith("_"):
                            # considers member as public
                            public.append(name)
        return public, items

    def get_module_attrs(members: Any) -> Tuple[List[str], List[str]]:
        """Find module attributes with docstrings."""
        attrs, public = [], []
        try:
            analyzer = ModuleAnalyzer.for_module(name)
            attr_docs = analyzer.find_attr_docs()
            for namespace, attr_name in attr_docs:
                if namespace == "" and attr_name in members:
                    attrs.append(attr_name)
                    if not attr_name.startswith("_"):
                        public.append(attr_name)
        except PycodeError:
            pass  # give up if ModuleAnalyzer fails to parse code
        return public, attrs

    def get_modules(obj: Any) -> Tuple[List[str], List[str]]:
        items: List[str] = []
        for _, modname, ispkg in pkgutil.iter_modules(obj.__path__):
            fullname = name + "." + modname
            try:
                module = import_module(fullname)
                if module and hasattr(module, "__sphinx_mock__"):
                    continue
            except ImportError:
                pass

            items.append(fullname)
        public = [x for x in items if not x.split(".")[-1].startswith("_")]
        return public, items

    ns: Dict[str, Any] = {}
    ns.update(context)

    if doc.objtype == "module":
        scanner = ModuleScanner(app, obj)
        ns["members"] = scanner.scan(imported_members)
        ns["functions"], ns["all_functions"] = get_members(
            obj, {"function"}, imported=imported_members
        )
        ns["classes"], ns["all_classes"] = get_members(obj, {"class"}, imported=imported_members)
        ns["exceptions"], ns["all_exceptions"] = get_members(
            obj, {"exception"}, imported=imported_members
        )
        ns["attributes"], ns["all_attributes"] = get_module_attrs(ns["members"])
        ispackage = hasattr(obj, "__path__")
        if ispackage and recursive:
            ns["modules"], ns["all_modules"] = get_modules(obj)
    elif doc.objtype == "class":
        ns["members"] = dir(obj)
        ns["inherited_members"] = set(dir(obj)) - set(obj.__dict__.keys())
        ns["methods"], ns["all_methods"] = get_members(obj, {"method"}, ["__init__"])
        ns["attributes"], ns["all_attributes"] = get_members(obj, {"attribute", "property"})

    if modname is None or qualname is None:
        modname, qualname = split_full_qualified_name(name)

    if doc.objtype in ("method", "attribute", "property"):
        ns["class"] = qualname.rsplit(".", 1)[0]

    if doc.objtype in ("class",):
        shortname = qualname
    else:
        shortname = qualname.rsplit(".", 1)[-1]

    ns["fullname"] = name
    ns["module"] = modname
    ns["objname"] = qualname
    ns["name"] = shortname

    ns["objtype"] = doc.objtype
    ns["underline"] = len(name) * "="

    if template_name:
        return template.render(template_name, ns)
    else:
        return template.render(doc.objtype, ns)
