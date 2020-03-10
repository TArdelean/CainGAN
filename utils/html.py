from utils.util import opt
import dominate
from dominate.tags import *
import os


# Borrows from https://github.com/NVIDIA/pix2pixHD
class HTML:
    def __init__(self, web_dir, title, refresh=0):
        self.title = title
        self.web_dir = web_dir
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=None):
        if width is None:
            width = opt.image_size[0]
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            self.link_image(link, width)
                            br()
                            p(txt)

    def link_image(self, link, width):
        with a(href=os.path.join('../../../', link)):
            img(style="width:%dpx" % (width), src=os.path.join('../../../', link))

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()
