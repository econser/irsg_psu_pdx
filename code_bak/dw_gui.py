#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

"""
    Expects ./viz_config.json for setting the base image directory
"""


from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog, tkFont
from PIL import ImageTk, Image, ImageDraw, ImageFont
import os.path
import csv, json
import random, colorsys

image_basedir = './'
try:
    cfg_file = open('viz_config.json')
    if cfg_file is not None:
        cfg_data = json.load(cfg_file)
        image_basedir = cfg_data['image_basedir']
except:
    pass

class ClassificationBox:
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class Classification:
    def __init__(self, image_id=None, query=None, objects=None, energy=None, image_class=None, image_subdir=None, image_filename=None):
        self.query = query
        if objects is None:
            self.objects = []
        else:
            self.objects = objects
        self.energy = energy
        self.image_class = image_class
        self.image_id = image_id
        self.image_subdir = image_subdir
        self.image_filename = image_filename

class simpleapp_tk(Tk):
    def __init__(self, parent):
        Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.menu = Frame(bg='lightgray', bd=0)
        self.menu.pack(fill=X)
        self.menu2 = Frame(bg='lightgray', bd=0)
        self.menu2.pack(fill=X)

        self.filenameLabel = Label(self.menu, text='Datafile', bg='lightgray', fg='black')
        self.filenameLabel.pack(side=LEFT)
        self.filenameVariable = StringVar()
        self.filenameVariable.trace("w", lambda name, index, mode, sv=self.filenameVariable: self.process_filename())
        self.filenameEntry = Entry(self.menu, textvariable=self.filenameVariable, highlightbackground='lightgray')
        self.filenameEntry.pack(side=LEFT)
        self.filenameBrowseButton = Button(self.menu, text='Browse', highlightbackground='lightgray', command=self.update_filename)
        self.filenameBrowseButton.pack(side=LEFT)

        self.filenameChanged = True
        self.fileData = []
        self.imageIdToFilename = dict()
        #imageJsonFile = json.load(open("./sg_data/sg_test_annotations.json"))
        #for index, elem in enumerate(imageJsonFile):
        #    self.imageIdToFilename[index] = elem['filename']

        self.filenameImageIdSeparator = Label(self.menu, text=' ', bg='lightgray', fg='black')
        self.imageIdLabel = Label(self.menu, text='Image ID', bg='lightgray', fg='black')
        self.imageIdVariable = IntVar()
        self.imageIdEntry = Entry(self.menu, textvariable=self.imageIdVariable, w=4, highlightbackground='lightgray')
        self.loadImageButton = Button(self.menu, text='Open Image', highlightbackground='lightgray', command=self.process_image_id)
        self.exportImageButton = Button(self.menu, text='Export PNG', highlightbackground='lightgray', command=self.export_canvas)
        self.queryLabel = Label(self.menu2, text='No loaded image', bg='lightgray', fg='black')

        self.frame = Frame(bg='gray')
        self.frame.pack(fill=BOTH, expand=True)

        self.frameCanvas = Canvas(self.frame, bg='gray', highlightbackground='gray')
        self.frameCanvas.pack(fill=BOTH, expand=True)

        self.referenceImage = None
        self.referenceImageAspectRatio = float(self.referenceImage.height) / float(self.referenceImage.width) if self.referenceImage is not None else None
        self.renderedImage = None
        self.canvasImage = None
        self.validImageIds = []
        self.imageIdsByEnergy = []
        self.imageIdsIndex = 0

        self.frameCanvas.bind("<Configure>", self.resize_frame_event)
        self.imageIdEntry.bind("<Return>", lambda event: self.process_image_id())
        self.bind("<Right>", lambda event: self.next_image_id())
        self.bind("<Left>", lambda event: self.prev_image_id())

        self.classificationBoxes = []

    def hide_image_id_menu(self):
        self.filenameImageIdSeparator.pack_forget()
        self.imageIdLabel.pack_forget()
        self.imageIdEntry.pack_forget()
        self.loadImageButton.pack_forget()
        self.queryLabel.pack_forget()
        self.exportImageButton.pack_forget()
        self.queryLabel.config(text='No loaded image')

    def show_and_reset_id_menu(self):
        self.filenameImageIdSeparator.pack(side=LEFT)
        self.imageIdLabel.pack(side=LEFT)
        self.imageIdEntry.pack(side=LEFT)
        self.loadImageButton.pack(side=LEFT)
        self.queryLabel.pack(side=LEFT)
        self.exportImageButton.pack(side=LEFT)
        self.imageIdVariable.set(0)

    def update_filename(self):
        filename = tkFileDialog.askopenfilename(initialdir='.', title='Select file', filetypes=(("csv files", "*.csv"),))
        if filename != '':
            self.filenameVariable.set(filename)

    def process_filename(self):
        self.filenameChanged = True
        filename = self.filenameVariable.get()
        if filename.endswith('.csv') and os.path.isfile(filename):
            self.filenameEntry.config(bg='white')
            self.show_and_reset_id_menu()
        else:
            self.filenameEntry.config(bg='red')
            self.hide_image_id_menu()

    def process_image_id(self):
        if self.filenameChanged:
            self.load_config()
            classifications = dict()
            validImageIds = set()
            for line in self.fileData:
                image_id = int(line[0])
                data_type = int(line[1])
                validImageIds.add(image_id)
                classification = classifications.get(image_id, Classification(image_id=image_id))
                if data_type == 0:
                    classification.query = line[2][2:-1]
                elif data_type == 1:
                    classification.objects.append(ClassificationBox(line[3][2:-1], int(line[4]), int(line[5]), int(line[6]), int(line[7])))
                elif data_type == 2:
                    classification.energy = float(line[2])
                elif data_type == 3:
                    classification.image_class = line[2][2:-1]
                elif data_type == 4:
                    classification.image_subdir = line[2][1:]
                elif data_type == 5:
                    classification.image_filename = line[2][1:]
                classifications[image_id] = classification
            self.classifications = classifications
            self.validImageIds = sorted(list(validImageIds))
            self.imageIdsByEnergy = sorted(map(lambda x: (x.energy, x.image_id), self.classifications.values()))
            self.imageIdsIndex = 0
            self.filenameChanged = False
            self.imageIdVariable.set(self.imageIdsByEnergy[self.imageIdsIndex][1])
        targetId = self.imageIdVariable.get()

        self.classificationBoxes = self.classifications[targetId].objects
        self.queryLabel.config(text=(self.classifications[targetId].query + ' : ' + self.classifications[targetId].image_class + " : " + str(self.classifications[targetId].energy)))
        #self.referenceImage = Image.open('./sg_data/images/' + self.imageIdToFilename[targetId])
        self.referenceImage = Image.open(image_basedir + self.classifications[targetId].image_subdir + '/' + self.classifications[targetId].image_filename)
        self.referenceImageAspectRatio = float(self.referenceImage.height) / float(self.referenceImage.width) if self.referenceImage is not None else None
        self.resize_frame(self.frameCanvas.winfo_width(), self.frameCanvas.winfo_height())

    def export_canvas(self):
        if self.referenceImage is not None:
            filename = tkFileDialog.asksaveasfilename(initialdir='.', title='Select file', filetypes=(("jpeg files", "*.jpg"),))
            if len(filename) < 4 or filename[:-4] != ".png":
                filename = filename + ".png"

            refWidth = self.referenceImage.width
            refHeight = self.referenceImage.height
            self.classificationBoxes.sort(key=lambda b: (1 + b.y) / (refWidth - (b.x + (b.w / 2.0))))

            font = tkFont.Font(family="Helvetica", size=20)
            renderedFont = ImageFont.truetype("./Helvetica.ttf", 20)
            labelMaxWidthLeft = 0
            labelMaxWidthRight = 0
            for box in self.classificationBoxes:
                leftSide = box.x + (box.w / 2.0) < self.referenceImage.width / 2.0
                labelWidth, _ = renderedFont.getsize(box.text + '--')
                if leftSide:
                    if labelMaxWidthLeft < labelWidth:
                        labelMaxWidthLeft = labelWidth
                else:
                    if labelMaxWidthRight < labelWidth:
                        labelMaxWidthRight = labelWidth

            labelMargin = 8
            image = Image.new("RGB", (self.referenceImage.width + labelMaxWidthLeft + labelMaxWidthRight + labelMargin * 2, self.referenceImage.height), (255, 255, 255))
            image.paste(self.referenceImage, (labelMaxWidthLeft + labelMargin, 0))
            draw = ImageDraw.Draw(image)
            leftVerticalPositionAccumulator = 0
            rightVerticalPositionAccumulator = 0
            _, heightDifferential = renderedFont.getsize("Lorem ipsum")
            for box in self.classificationBoxes:
                r, g, b = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
                lineWidth = 4
                for i in range(lineWidth):
                    draw.rectangle((int(labelMaxWidthLeft + labelMargin + box.x)+i,
                        int(box.y)+i,
                        int(labelMaxWidthLeft + labelMargin + (box.x+box.w))+i,
                        int((box.y+box.h))+i), outline=(int(r*255), int(g*255), int(b*255)))

                leftSide = box.x + (box.w / 2.0) < self.referenceImage.width / 2.0
                if leftSide:
                    fontWidth, _ = renderedFont.getsize(box.text)
                    draw.text((labelMaxWidthLeft + labelMargin / 2 - fontWidth, leftVerticalPositionAccumulator),
                        box.text, fill=(0, 0, 0), font=renderedFont)
                    draw.line((labelMaxWidthLeft + labelMargin + int((box.x+box.x+box.w) / 2),
                        int(box.y),
                        labelMaxWidthLeft + labelMargin,
                        leftVerticalPositionAccumulator + heightDifferential / 2), fill=(int(r*255), int(g*255), int(b*255)), width=lineWidth)
                    leftVerticalPositionAccumulator += heightDifferential
                else:
                    draw.text((self.referenceImage.width + labelMaxWidthLeft + (3 * labelMargin / 2),
                        rightVerticalPositionAccumulator), box.text, fill=(0, 0, 0), font=renderedFont)
                    draw.line((labelMaxWidthLeft + labelMargin + int((box.x+box.x+box.w) / 2),
                        int(box.y),
                        self.referenceImage.width + labelMaxWidthLeft + labelMargin,
                        rightVerticalPositionAccumulator + heightDifferential / 2), fill=(int(r*255), int(g*255), int(b*255)), width=lineWidth)
                    rightVerticalPositionAccumulator += heightDifferential
            image.save(filename, "PNG")


    def load_config(self):
        filename = self.filenameVariable.get()
        self.fileData = []
        if filename.endswith('.csv') and os.path.isfile(filename):
            with open(filename) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
                for line in csvreader:
                    self.fileData.append(line)

    def resize_frame(self, width, height):
        if width <= 2:
            return
        if height <=0:
            return
        if self.referenceImage is not None:
            if self.canvasImage is not None:
                self.frameCanvas.delete(ALL)

            refWidth = self.referenceImage.width
            refHeight = self.referenceImage.height
            self.classificationBoxes.sort(key=lambda b: (1 + b.y) / (refWidth - (b.x + (b.w / 2.0))))

            font = tkFont.Font(family="Helvetica", size=20)
            labelMaxWidthLeft = 0
            labelMaxWidthRight = 0
            for box in self.classificationBoxes:
                leftSide = box.x + (box.w / 2.0) < self.referenceImage.width / 2.0
                labelWidth = font.measure(box.text + '--')
                if leftSide:
                    if labelMaxWidthLeft < labelWidth:
                        labelMaxWidthLeft = labelWidth
                else:
                    if labelMaxWidthRight < labelWidth:
                        labelMaxWidthRight = labelWidth

            globalScaleModifier = 1
            labelMargin = 8
            if float(width-(labelMaxWidthRight+labelMaxWidthLeft+labelMargin+labelMargin)) / float(height) < float(self.referenceImage.width) / float(self.referenceImage.height):
                scaleModifier = (float(width) - float(labelMaxWidthRight+labelMaxWidthLeft+labelMargin+labelMargin)) / float(width)
                globalScaleModifier = width*scaleModifier / self.referenceImage.width
                self.renderedImage = ImageTk.PhotoImage(self.referenceImage.resize((int(width*scaleModifier), int(width*scaleModifier*self.referenceImageAspectRatio)), Image.NEAREST))
            else:
                attemptedWidth = height / self.referenceImageAspectRatio
                globalScaleModifier = attemptedWidth / self.referenceImage.width
                self.renderedImage = ImageTk.PhotoImage(self.referenceImage.resize((int(attemptedWidth), int(attemptedWidth*self.referenceImageAspectRatio)), Image.NEAREST))
            self.canvasImage = self.frameCanvas.create_image(labelMaxWidthLeft + labelMargin, 0, image=self.renderedImage, anchor='nw')
            leftVerticalPositionAccumulator = 0
            rightVerticalPositionAccumulator = 0
            heightDifferential = font.metrics("linespace")
            for box in self.classificationBoxes:
                r, g, b = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
                fill = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
                lineWidth = 4
                self.frameCanvas.create_rectangle(int(labelMaxWidthLeft + labelMargin + box.x * globalScaleModifier),
                    int(box.y * globalScaleModifier),
                    int(labelMaxWidthLeft + labelMargin + (box.x+box.w) * globalScaleModifier),
                    int((box.y+box.h) * globalScaleModifier), fill='', outline=fill, width=lineWidth)

                leftSide = box.x + (box.w / 2.0) < self.referenceImage.width / 2.0
                if leftSide:
                    self.frameCanvas.create_text(labelMaxWidthLeft + labelMargin / 2, leftVerticalPositionAccumulator,
                        text=box.text, anchor="ne", fill='black', font=font)
                    self.frameCanvas.create_line(labelMaxWidthLeft + labelMargin + int((box.x+box.x+box.w) * globalScaleModifier / 2),
                        int(box.y * globalScaleModifier),
                        labelMaxWidthLeft + labelMargin,
                        leftVerticalPositionAccumulator + heightDifferential / 2, fill=fill, width=lineWidth)
                    leftVerticalPositionAccumulator += heightDifferential
                else:
                    self.frameCanvas.create_text(self.renderedImage.width() + labelMaxWidthLeft + (3 * labelMargin / 2),
                        rightVerticalPositionAccumulator, text=box.text, anchor="nw", fill='black', font=font)
                    self.frameCanvas.create_line(labelMaxWidthLeft + labelMargin + int((box.x+box.x+box.w) * globalScaleModifier / 2),
                        int(box.y * globalScaleModifier),
                        self.renderedImage.width() + labelMaxWidthLeft + labelMargin,
                        rightVerticalPositionAccumulator + heightDifferential / 2, fill=fill, width=lineWidth)
                    rightVerticalPositionAccumulator += heightDifferential


    def resize_frame_event(self, event):
        self.resize_frame(event.width, event.height)

    def next_image_id(self):
        self.imageIdsIndex = (self.imageIdsIndex + 1) % len(self.imageIdsByEnergy)
        self.imageIdVariable.set(self.imageIdsByEnergy[self.imageIdsIndex][1])
        self.process_image_id()

    def prev_image_id(self):
        self.imageIdsIndex = (self.imageIdsIndex - 1) % len(self.imageIdsByEnergy)
        self.imageIdVariable.set(self.imageIdsByEnergy[self.imageIdsIndex][1])
        self.process_image_id()

if __name__ == '__main__':
    app = simpleapp_tk(None)
    app.title('sg viz tool')
    app.mainloop()
