"""
Sphinx plugin to run generate a gallery for notebooks

Modified from the seaborn project, which modified the mpld3 project.
"""
import base64
import json
import os
import glob
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import image

DOC_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_OF_CONTENTS_FILENAME = "table_of_contents.js"

INDEX_TEMPLATE = """
.. _{sphinx_tag}:

.. title:: example_notebooks

.. raw:: html

    <h1 class="ui header">Example Notebooks</h1>
    <div id="gallery" class="ui vertical segment">
    </div>
"""


def create_thumbnail(infile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    """Overwrites `infile` with a new file of the given size"""
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cx * rows - size // 2)), rows - size)
        yslice = slice(ymin, ymin + size)
    else:
        yslice = slice(0, size)
        xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
        xslice = slice(xmin, xmin + size)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(infile, dpi=dpi)
    plt.close(fig)
    return fig


class NotebookGenerator(object):
    """Tools for generating an example page from a file"""

    def __init__(self, filename, target_dir):
        self.basename = os.path.basename(filename)
        self.stripped_name = os.path.splitext(self.basename)[0]
        self.output_html = os.path.join(
            "..", "notebooks", "{}.html".format(self.stripped_name)
        )
        self.image_dir = os.path.join(target_dir, "_images")
        self.png_path = os.path.join(
            self.image_dir, "{}.png".format(self.stripped_name)
        )
        with open(filename, "r") as fid:
            self.json_source = json.load(fid)
        self.pagetitle = self.extract_title()
        self.default_image_loc = os.path.join(
            os.path.dirname(DOC_SRC), "logos", "PyMC3.png"
        )

        # Only actually run it if the output RST file doesn't
        # exist or it was modified less recently than the example
        if not os.path.exists(self.output_html) or (
            os.path.getmtime(self.output_html) < os.path.getmtime(filename)
        ):

            self.gen_previews()
        else:
            print("skipping {0}".format(filename))

    def extract_preview_pic(self):
        """By default, just uses the last image in the notebook."""
        pic = None
        for cell in self.json_source["cells"]:
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", []):
                    pic = output["data"]["image/png"]
        if pic is not None:
            return base64.b64decode(pic)
        return None

    def extract_title(self):
        for cell in self.json_source["cells"]:
            if cell["cell_type"] == "markdown":
                rows = [row.strip() for row in cell["source"] if row.strip()]
                for row in rows:
                    if row.startswith("# "):
                        return row[2:]
        return self.basename.replace("_", " ")

    def gen_previews(self):
        preview = self.extract_preview_pic()
        if preview is not None:
            with open(self.png_path, "wb") as buff:
                buff.write(preview)
        else:
            shutil.copy(self.default_image_loc, self.png_path)
        create_thumbnail(self.png_path)


def main(app):
    working_dir = os.getcwd()
    os.chdir(app.builder.srcdir)
    static_dir = os.path.join(app.builder.srcdir, "_static")
    target_dir = os.path.join(app.builder.srcdir, "nb_examples")
    image_dir = os.path.join(target_dir, "_images")
    source_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(app.builder.srcdir)), "notebooks")
    )

    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    if not os.path.exists(source_dir):
        os.makedirs(source_dir)

    # Write individual example files
    data = {}
    for filename in sorted(glob.glob(os.path.join(source_dir, "*.ipynb"))):
        ex = NotebookGenerator(filename, target_dir)
        data[ex.stripped_name] = {
            "title": ex.pagetitle,
            "url": os.path.join("/examples", ex.output_html),
            "thumb": os.path.basename(ex.png_path),
        }

    js_file = os.path.join(image_dir, "gallery_contents.js")
    with open(os.path.join(source_dir, TABLE_OF_CONTENTS_FILENAME), "r") as toc:
        table_of_contents = toc.read()

    js_contents = "Gallery.examples = {}\n{}".format(
        json.dumps(data), table_of_contents
    )

    with open(js_file, "w") as js:
        js.write(js_contents)

    with open(os.path.join(target_dir, "index.rst"), "w") as index:
        index.write(INDEX_TEMPLATE.format(sphinx_tag="notebook_gallery"))

    os.chdir(working_dir)


def setup(app):
    app.connect("builder-inited", main)
