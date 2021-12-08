import sys
import random
from copy import deepcopy
import multiprocessing
import jsonpickle
import re
import numpy
from PIL import Image, ImageDraw
import numpy as np
import cv2
import glob
import decimal
import os

POP_PER_GENERATION = 20
MUTATION_CHANCE = 0.02
ADD_GENE_CHANCE = 0.3
REM_GENE_CHANCE = 0.2
INITIAL_GENES = 50
#How often to output images and save files
GENERATIONS_PER_IMAGE = 50
GENERATIONS_PER_SAVE = 50
SCRAP_NUMBER = 300
scraps = []



def has_transparency(img):
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True

    return False

def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src

def crop_this(image_file, start_row, start_column, length, width, with_plot=False, gray_scale=False):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    image_shape = image_src.shape
    
    length = abs(length)
    width = abs(width)
    
    start_row = start_row if start_row >= 0 else 0
    start_column = start_column if start_column >= 0 else 0
    
    end_row = length + start_row
    end_row = end_row if end_row <= image_shape[0] else image_shape[0]
    
    end_column = width + start_column
    end_column = end_column if end_column <= image_shape[1] else image_shape[1]
    
    image_cropped = image_src[start_row:end_row, start_column:end_column]
    cmap_val = None if not gray_scale else 'gray'
    
    if with_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("Cropped")
        
        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_cropped, cmap=cmap_val)
        return True
    return image_cropped

def getScraps(target):
    print "INITIALISING SCRAPS..."
    canvaswidth, canvasheight = target.size

    for x in range(0, SCRAP_NUMBER):
        
        album = random.choice(glob.glob(r"yourpathhere\albums\*.png"))
          
        pilalbum = Image.open(album)
        if has_transparency(pilalbum):
            
            albumwidth, albumheight = pilalbum.size
           
            scalar = float(decimal.Decimal(random.randrange(1, 75))/100)
        
            color = pilalbum.resize((int(albumwidth*scalar), int(albumheight*scalar)))
            mwidth, height = color.size
            scraps.append(numpy.array(color))
        else:
            pass
        

    print "...SCRAPS INITIALISIED"

try:
    globalTarget = Image.open("reference.png")
    globalTarget = globalTarget.convert('RGBA')
    
    print(has_transparency(globalTarget))

    
    getScraps(globalTarget)
except IOError:
    print "File reference.png must be located in the same directory."
    exit()
#-------------------------------------------------------------------------------------------------
#Helper Classes
#-------------------------------------------------------------------------------------------------
class Point:
    """
    A 2D point. You can add them together if you want.
    """
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __add__(self,o):
        return Point(self.x+o.x,self.y+o.y)

class Color:
    """
    A color. You can shift it by a given value.
    """
    def __init__(self,r,g,b,a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def shift(self,r,g,b):
        self.r = max(0,min(255,self.r+r))
        self.g = max(0,min(255,self.g+g))
        self.b = max(0,min(255,self.b+b))
        self.a = max(0,min(255,self.a+a))

    def __str__(self):
        return "({},{},{},{})".format(self.r,self.g,self.b,self.a)

#-------------------------------------------------------------------------------------------------
#Genetic Classes
#-------------------------------------------------------------------------------------------------
class Gene:

    def __init__(self,size):
        self.size = size #The canvas size so we know the maximum position value

        #self.diameter = random.randint(5,15)
        
        self.position = Point(random.randint(20,size[0]-20),random.randint(20,size[1]-20))
        self.color = random.choice(scraps)
        self.params = ["position", "color", "rotation"]
        self.rotation = random.randint(0, 360)
        self.width = self.color.shape[0] 
        self.length = self.color.shape[1]
        
    def mutate(self):
        #Decide how big the mutation will be
        mutation_size = max(1,int(round(random.gauss(15,4))))/100

        #Decide what will be mutated
        

        mutation_type = random.choice(self.params)

        #Mutate the thing

        if mutation_type == "position":
            x = max(0,random.randint(int(self.position.x*(1-mutation_size)),int(self.position.x*(1+mutation_size))))
            y = max(0,random.randint(int(self.position.y*(1-mutation_size)),int(self.position.y*(1+mutation_size))))
            self.position = Point(min(x,self.size[0]),min(y,self.size[1]))

        elif mutation_type == "color":        
            self.color = random.choice(scraps)

        elif mutation_type == "rotation":        
            self.rotation = random.randint(int(self.rotation*(1-mutation_size)),int(self.rotation(1+mutation_size)))

       
    def mutatetwo(self):
        #Decide how big the mutation will be
        #Decide what will be mutated
        

        mutation_type = "position"

        #Mutate the thing

        if mutation_type == "position":
            x = random.randint(self.size[0]*0.4, self.size[0]*0.6)
            y = random.randint(self.size[1]*0.4, self.size[1]*0.6)
            self.position = Point(x, y)

        else:
            pass



            


    def getSave(self):
        """
        Allows us to save an individual gene in case the program is stopped.
        """
        so = {}
        
        so["position"] = Point(self.position.x,self.position.y)
        so["color"] = self.color
        so["rotation"] = self.rotation
        
        return so

    def loadSave(self,so):
        """
        Allows us to load an individual gene in case the program is stopped.
        """
        #self.size = so["size"]
        #self.diameter = so["diameter"]
        #self.pos = Point(so["pos"][0],so["pos"][1])
        #self.color = Color(so["color"][0],so["color"][1],so["color"][2])

        self.position = Point(so["position"][0],so["position"][1])
        self.color = so["color"]
        self.rotation = so["rotation"]
        

class Organism:

    def __init__(self,size,num):
        self.size = size

        #Create random genes up to the number given
        self.genes = [Gene(size) for _ in xrange (num)]        


    def mutate(self):
        #For small numbers of genes, each one has a random chance of mutating
        try:
            if len(self.genes) < 200:
                for g in self.genes:
                    if MUTATION_CHANCE < random.random():
                        g.mutate()

            #For large numbers of genes, pick a random sample, this is statistically equivalent and faster
            else:
                for g in random.sample(self.genes,int(len(self.genes)*MUTATION_CHANCE)):
                    g.mutate()
        except:
            pass

        #We also have a chance to add or remove a gene
        try:
            if ADD_GENE_CHANCE < random.random():
                self.genes.append(Gene(self.size))
            if len(self.genes) > 0 and REM_GENE_CHANCE < random.random():
                self.genes.remove(random.choice(self.genes))
        except:
            pass

    def mutatetwo(self):
        #For small numbers of genes, each one has a random chance of mutating
        try:
            if len(self.genes) < 200:
                for g in self.genes:
                    if MUTATION_CHANCE < random.random():
                        g.mutatetwo()

            #For large numbers of genes, pick a random sample, this is statistically equivalent and faster
            else:
                for g in random.sample(self.genes,int(len(self.genes)*MUTATION_CHANCE)):
                    g.mutatetwo()
        except:
            pass

        #We also have a chance to add or remove a gene
        try:
            if ADD_GENE_CHANCE < random.random():
                self.genes.append(Gene(self.size))
            if len(self.genes) > 0 and REM_GENE_CHANCE < random.random():
                self.genes.remove(random.choice(self.genes))
        except:
            pass

    def drawImage(self):
        """
        Using the Image module, use the genes to draw the image.
        """
        image = Image.new("RGBA",self.size,(255,255,255,0))
        #canvas = ImageDraw.Draw(image)

        for g in self.genes:
            #print(g.color.shape)


            try:
                paper = Image.fromarray(g.color)
                
                #canvas.ellipse([g.pos.x-g.diameter,g.pos.y-g.diameter,g.pos.x+g.diameter,g.pos.y+g.diameter],outline=color,fill=color)
                x = g.position.x
                y = g.position.y
                
                #paper = paper.convert('RGBA')
                paper = paper.rotate(g.rotation, expand=1)
                image.paste(paper, (x, y), paper)
            except:
                print "Genetic reset"
                
                paper = Image.fromarray(g.color)
                
                #canvas.ellipse([g.pos.x-g.diameter,g.pos.y-g.diameter,g.pos.x+g.diameter,g.pos.y+g.diameter],outline=color,fill=color)
                
                x = random.randint(g.size[0]*0.2, gisize[0]*0.8)
                y = random.randint(g.size[1]*0.2, gisize[1]*0.8)
                
                #paper = paper.convert('RGBA')
                paper = paper.rotate(g.rotation, expand=1)
                image.paste(paper, (int(x*0.50), int(y*0.50)), paper)

        return image

    def getSave(self,generation):
        """
        Allows us to save an individual organism in case the program is stopped.
        """
        so = [generation]
        return so + [g.getSave() for g in self.genes]

    def loadSave(self,so):
        """
        Allows us to load an individual organism in case the program is stopped.
        """
        self.genes = []
        gen = so[0]
        so = so[1:]
        for g in so:
            newGene = Gene(self.size)
            newGene.loadSave(g)
            self.genes.append(newGene)
        return gen

def fitness(im1,im2):



    """
    The fitness function is used by the genetic algorithm to determine how successful a given organism
    is. Usually a genetic algorithm is trying to either minimize or maximize this function.

    This one uses numpy to quickly compute the sum of the differences between the pixels.
    """
    #Convert Image types to numpy arrays
    i1 = numpy.array(im1,numpy.int16)
    i2 = numpy.array(im2,numpy.int16)

    i1 = i1[:,:,:3]
    i2 = i2[:,:,:3]

    dif = numpy.sum(numpy.abs(i1-i2))
    return (dif / 255.0 * 100) / i1.size

#-------------------------------------------------------------------------------------------------
#Functions to Make Stuff Run
#-------------------------------------------------------------------------------------------------
def run(cores,so=None):
    """
    Contains the loop that creates and tests new generations.
    """
    #Create storage directory in current directory
    if not os.path.exists("results"):
        os.mkdir("results")

    #Create output log file
    f = file(os.path.join("results","log.txt"),'a')

    target = globalTarget


    #Create the parent organism (with random genes)
    generation = 1
    parent = Organism(target.size,INITIAL_GENES)

    #Load the save if one is given
    if so != None:
        gen = parent.loadSave(jsonpickle.decode(so))
        generation = int(gen)
    prevScore = 101
    score = fitness(parent.drawImage(),target)
    #Setup the multiprocessing pool
    p = multiprocessing.Pool(cores)
    #Infinite loop (until the process is interrupted)
    while True:
        #Print the current score and write it to the log file
        print "Generation {} - {}".format(generation,score)
        f.write("Generation {} - {}\n".format(generation,score))

        #Save an image of the current best organism to the results directory
        if (generation) % GENERATIONS_PER_IMAGE == 0:
            parent.drawImage().save(os.path.join("results","{}.png".format(generation)))
        generation += 1
        
        prevScore = score

        #Spawn children
        children = []
        scores = []

        #Keep the best from before in case all mutations are bad
        children.append(parent)
        scores.append(score)

        #Perform the mutations and add to the parent
        try:
            results = groupMutate(parent,POP_PER_GENERATION-1,p)
        except KeyboardInterrupt:
            print 'Bye!'
            p.close()
            return
        
        newScores,newChildren = zip(*results)

        children.extend(newChildren)
        scores.extend(newScores)
        #Find the winner

        winners = sorted(zip(children,scores),key=lambda x: x[1])

        parent,score = winners[0]


        #Store a backup to resume running if the program is interrupted
        if generation % 100 == 0:
            sf = file(os.path.join("results","{}.txt".format(generation)),'w')
            try:
                sf.write(jsonpickle.encode(parent.getSave(generation)))
            except:
                print 'error'
            sf.close()

def mutateAndTest(o):
    """
    Given an organism, perform a random mutation on it, and then use the fitness function to
    determine how accurate of a result the mutated offspring draws.
    """    

    try:
        c = deepcopy(o)
        c.mutate()

        i1 = c.drawImage()
        i2 = globalTarget
        return (fitness(i1,i2),c)
    
    except ValueError:
        try:
            print("Genetic Reset")
            c = deepcopy(o)
            c.mutatetwo()

            i1 = c.drawImage()
            i2 = globalTarget
            return (fitness(i1,i2),c)
        except:
            return (float(99.99),[[255,255,255], [255,255,255], [255,255,255]])
        


    """
    try:
        c = deepcopy(o)
        c.mutate()
        try:
            i1 = c.drawImage()
            i2 = globalTarget
            return (fitness(i1,i2),c)
        except:
            print "mutation failed"
            i1 = o.drawImage()
            i2 = globalTarget
            return (fitness(i1,i2),o)
        
    except KeyboardInterrupt, e:
        pass
    """

def groupMutate(o,number,p):

    """
    Mutates and tests a number of organisms using the multiprocessing module.
    """
    
    results = p.map(mutateAndTest,[o]*int(number))
    
    return results
        
#-------------------------------------------------------------------------------------------------
#Main Function
#-------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    #Set defaults
    cores = max(1,multiprocessing.cpu_count())
    so = None

	#Check the arguments, options are currents -t (number of threads) and -s (save file)
    if len(sys.argv) > 1:
        args = sys.argv[1:]

        for i,a in enumerate(args):
            if a == "-t":
                cores = int(args[i+1])
            elif a == "-s":
                with open(args[i+1],'r') as save:
                    so = save.read()

    run(cores,so)
