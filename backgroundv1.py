import pygame, os
from pygame.locals import *
import numpy as np
import pickle

file = r'images\sheetx2.png'
#80 is grass
#230 is a crate
#72-77 is decoration
pixelsize = 32
#print(__file__)
path = os.path.abspath(__file__)
dir = os.path.dirname(path)

base = os.path.basename(path)
#print(base)

root, ext = os.path.splitext(path)
#print(root)
#print(os.getcwd())
p = root.replace(r"\backgroundv1","")
os.chdir(p)

middle = pygame.sprite.Group()
front = pygame.sprite.Group()

class Tileset:
    def __init__(self, file, size=(pixelsize, pixelsize), margin=0, spacing=0):
        self.file = file
        self.size = size
        self.margin = margin
        self.spacing = spacing
        self.image = pygame.image.load(file)
        self.rect = self.image.get_rect()
        self.tiles = []
        self.load()


    def load(self):

        self.tiles = []
        x0 = y0 = self.margin
        w, h = self.rect.size
        dx = self.size[0] + self.spacing
        dy = self.size[1] + self.spacing
        
        for x in range(x0, w, dx):
            for y in range(y0, h, dy):
                tile = pygame.Surface(self.size)
                tile.blit(self.image, (0, 0), (x, y, *self.size))
                self.tiles.append(tile)

    def __str__(self):
        return f'{self.__class__.__name__} file:{self.file} tile:{self.size}'


class Tilemap: #TODO: Auto sizing
    def __init__(self, tileset, size=(32, 32), rect=None):
        np.random.seed(4)
        self.size = size
        self.tileset = tileset
        self.map = np.zeros(size, dtype=int)

        h, w = self.size
        self.image = pygame.Surface((pixelsize*w, pixelsize*h))
        if rect:
            self.rect = pygame.Rect(rect)
        else:
            self.rect = self.image.get_rect()

        


    def render(self):
        m, n = self.map.shape
        for i in range(m):
            for j in range(n):
                tile = self.tileset.tiles[self.map[i, j]]
                self.image.blit(tile, (j*pixelsize, i*pixelsize))

    def set_map(self, replay, step):
        field = np.full((self.size),80) 
        hivelocation = np.where(replay[step][2] == 1)
        field[hivelocation[0],hivelocation[1]] = 4 #HIVE LOCATION
        
        flowerlocations = np.where(replay[step][1] == -1, -77,(replay[step][1]))
        flowerlocations = np.where(flowerlocations == 1, -80,flowerlocations)
        
        field = np.add(field, flowerlocations) 

        
        print(step)


        #print(self.map)
        #print(self.map.shape)
        self.map = field
        self.render()

    

    def __str__(self):
        return f'{self.__class__.__name__} {self.size}'      

def load_episode(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
class Game:
    W = 1024
    H = 1024
    SIZE = W, H

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(Game.SIZE)
        pygame.display.set_caption('Pygame Tile Demo')
        self.running = True

        self.step = 0

        self.tileset = Tileset(file)
        self.tilemap = Tilemap(self.tileset)
        self.map2 = Tilemap(self.tileset)

    

    def run(self):
        self.replay = load_episode("episode1.pkl")
        bee = pygame.image.load(r'images\bee32.png')
        #print(replay[0][1].shape)
        
        print()
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

                elif event.type == KEYDOWN:
                    if event.key == K_m:
                        hive_level = self.replay[self.step][2]
                        
                        self.step += 1
                    elif event.key == K_c:
                        self.step += 1
                    elif event.key == K_z:
                        self.tilemap.set_map(self.replay, self.step)
                    elif event.key == K_s:
                        self.save_image()
                        
            self.screen.blit(self.tilemap.image, self.tilemap.rect)
            b = {k: v for k, v in self.replay[self.step][0].items() if v}
            #b1 = {k: v for k, v in self.replay[self.step + 1][0].items() if v}
            
            #if b == b1:
            #    print("same")
            #else:
            #    print("false")

            for positions in b.keys():
                x, y = positions
                #print(x,y)
                sprite = bee
                sprite_rect = sprite.get_rect(center=((x + 0.5) * 32, (y + 0.5) * 32))
                self.screen.blit(sprite, sprite_rect)
            #pygame.display.update()
            pygame.display.flip()
        pygame.quit()

    def save_image(self):
        # Save a screen shot.

        path = os.path.abspath(__file__)
        head, tail = os.path.split(path)
        root, ext = os.path.splitext(path)
        pygame.image.save(self.screen, root +str(self.step) +'.png')

game = Game()
game.run()