import pygame
import random
import math
import os
import json
import sys
# load hình ảnh vào pygame
def load_images(filepaths,is2x = False):
    images = []
    for filepath in filepaths:
        image = pygame.image.load(filepath)
        if is2x:
            image = pygame.transform.scale2x(image)
        images.append(image)
    return images

# ---------------------------------------init game setting
screen_width = 1280
screen_height = 820
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Gold Miner Classic")

# ---------------------------------------init game entities
#init Text Game
text_game_image = pygame.image.load("./assets/images/text_game.png")
#init Gold
gold_image = pygame.image.load("./assets/images/gold.png")
#init Rock
rock_image = pygame.image.load("./assets/images/rock.png")
#init Mole
mole_image = pygame.image.load("./assets/images/mole.png")
#init Mole with Diamond
mole2_image = pygame.image.load("./assets/images/moleDiamond.png")
#init Skull
skull_image = pygame.image.load("./assets/images/skull.png")
#init Bone
bone_image = pygame.image.load("./assets/images/bone.png")
#init Diamond
diamond_image = pygame.image.load("./assets/images/diamond.png")
#init TNT
tnt_image = pygame.image.load("./assets/images/tnt.png")
# empty image
empty = pygame.image.load('./assets/images/empty.png')
# Question Bag
questionBag = pygame.image.load('./assets/images/question_bag.png')
#dynamite
dynamite_image = pygame.image.load('./assets/images/dynamite.png')
#init miner
miner_files = [
    "./assets/images/miner_01.png",
    "./assets/images/miner_02.png",
    "./assets/images/miner_03.png",
    "./assets/images/miner_04.png",
    "./assets/images/miner_05.png",
    "./assets/images/miner_06.png",
    "./assets/images/miner_07.png",
    "./assets/images/miner_08.png"
]
miner_images = load_images(miner_files)
# shopkeeper
shopkeeper_files = [
    "./assets/images/shopkeeper_01.png",
    "./assets/images/shopkeeper_02.png"
]
shopkeeper_images = load_images(shopkeeper_files)
#init explosive
explosive_files = [
    "./assets/images/ex1.png",
    "./assets/images/ex2.png",
    "./assets/images/ex3.png",
    "./assets/images/ex4.png",
    "./assets/images/ex5.png",
    "./assets/images/ex6.png",
    "./assets/images/ex7.png",
    "./assets/images/ex8.png",
    "./assets/images/ex9.png"
]
explosive_images = load_images(explosive_files,True)

#init hoo
hoo_files = [
    "./assets/images/hoo_01.png",
    "./assets/images/hoo_02.png",
    "./assets/images/hoo_03.png"
]
hoo_images = load_images(hoo_files)
hight_score = pygame.image.load('./assets/images/hight_score.png')
panel_image = pygame.image.load('./assets/images/panel.png')
panel_image = pygame.transform.scale2x(panel_image)
table_image = pygame.image.load('./assets/images/shop_table.png')
dialog_image = pygame.image.load('./assets/images/ui_dialog.png')
dialog_image = pygame.transform.scale2x(dialog_image)
continue_img = pygame.image.load('./assets/images/continue.png')
#init shop item
rock_collectors_book = pygame.image.load('./assets/images/rock_collectors_book.png')
strength_drink = pygame.image.load('./assets/images/strength_drink.png')
gem_polish = pygame.image.load('./assets/images/gem_polish.png')
clover = pygame.image.load('./assets/images/clover.png')
dynamite_shop = pygame.image.load('./assets/images/dynamite_shop.png')
exit_image = pygame.image.load('./assets/images/exit.png')
next_image = pygame.image.load('./assets/images/next.png')
# ---------------------------------------init BG
bgA = pygame.image.load('./assets/images/bg_level_A.jpg').convert()
bgA = pygame.transform.scale2x(bgA)
bgB = pygame.image.load('./assets/images/bg_level_B.jpg').convert()
bgB = pygame.transform.scale2x(bgB)
bgC = pygame.image.load('./assets/images/bg_level_C.jpg').convert()
bgC = pygame.transform.scale2x(bgC)
bgD = pygame.image.load('./assets/images/bg_level_D.jpg').convert()
bgD = pygame.transform.scale2x(bgD)
bg_top = pygame.image.load('./assets/images/bg_top.png').convert()
cut_scene = pygame.image.load('./assets/images/cut_scene.jpg').convert()
miner_menu = pygame.image.load('./assets/images/miner_menu.png')
miner_menu_rect  = miner_menu.get_rect(bottomright=(screen_width,screen_height))
start_BG = pygame.image.load('./assets/images/start_BG.jpg')
store_BG = pygame.image.load('./assets/images/bg_shop.png')
backgrounds = [bgA, bgB, bgC, bgD]




# ---------------------------------------init sound
pygame.mixer.pre_init(frequency=11025, size=-16, channels=8, buffer=2048)
pygame.init()
explosive_sound = pygame.mixer.Sound('./assets/audios/explosive.wav')
goal_sound = pygame.mixer.Sound('./assets/audios/goal.wav')
grab_back_sound = pygame.mixer.Sound('./assets/audios/grab_back.wav')
grab_start_sound = pygame.mixer.Sound('./assets/audios/grab_start.wav')
hook_reset_sound = pygame.mixer.Sound('./assets/audios/hook_reset.wav')
high_value_sound = pygame.mixer.Sound('./assets/audios/high_value.wav')
normal_value_sound = pygame.mixer.Sound('./assets/audios/normal_value.wav')
money_sound = pygame.mixer.Sound('./assets/audios/money.wav')
made_goal_sound = pygame.mixer.Sound('./assets/audios/made_goal.wav')
MiniGold_point = 50
NormalGold_point  = 100
NormalGoldPlus_point = 250
BigGold_point = 500
MiniRock_point = 11
NormalRock_point = 20
BigRock_point = 100
Diamond_point = 600
Mole_point = 2
MoleWithDiamond_point = 602
Skull_point = 20
Bone_point = 7

# ---------------------------------------init game parameter
score = 0
goal = 650
goalAddOn = 275

def reset_game_state():
    """Reset tất cả state về giá trị ban đầu"""
    global score, goal, current_level, pause, start_time
    score = 0
    goal = 650
    current_level = 1
    pause = False
    start_time = None
    print("🔄 Game state reset to initial values")
def load_level_data():
    try:
        with open('./assets/levels/levels.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Levels.json not found, using empty data")
        return {}
    except Exception as e:
        print(f"Error loading levels.json: {e}")
        return {}
def get_score():
    return score
def set_score(new_score):
    global score
    score = new_score

def get_goal():
    return goal
def set_goal(new_goal):
    global goal
    goal = new_goal

pause = False
def get_pause():
    return pause
def set_pause(new_pause):
    global pause
    pause = new_pause

start_time = None
def get_time():
    return start_time

def set_time(new_pause):
    global start_time
    start_time = new_pause

current_level = 1
def get_level():
    return current_level
def set_level(new_level):
    global current_level
    current_level = new_level

# Đường dẫn đến file txt
high_score_file = "high_scores.txt"
high_scores = []