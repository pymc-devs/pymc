import copy

from typing import Any, Dict


class SettingsResetter:
    """Stores a copy of initial settings so they can be reset on call"""

    initial_settings: Dict[str, Any]

    def __init__(self, step_method: Any, *settings: str):
        try:
            self.initial_settings = {
                param: copy.deepcopy(getattr(step_method, param)) for param in settings
            }
        except AttributeError:
            raise SettingNotFoundInAttribute("check arguments for typos")

    def __call__(self, step_method: Any) -> Any:
        for param, initial_value in self.initial_settings.items():
            setattr(step_method, param, copy.deepcopy(initial_value))


class SettingNotFoundInAttribute(BaseException):
    pass
