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
dynamite_count = 0  # Thêm biến để lưu số lượng dynamite
game_speed = 1  # Tốc độ game: 1x, 2x, 5x, 10x
use_fixed_timestep = False  # True = fixed dt cho RL, False = real time cho game bình thường

def reset_game_state():
    """Reset tất cả state về giá trị ban đầu"""
    global score, goal, current_level, pause, start_time, dynamite_count, game_speed, scaled_time_offset, use_fixed_timestep
    score = 0
    goal = 650
    current_level = 1
    pause = False
    start_time = None
    dynamite_count = 0  # Reset số dynamite về 0
    game_speed = 1  # Reset tốc độ về 1x
    scaled_time_offset = 0  # Reset thời gian đã scale
    use_fixed_timestep = False  # Reset về real time cho game bình thường

def reset_level_state(keep_score=True, keep_dynamite=True):
    """Reset state cho level mới, giữ nguyên score và dynamite nếu cần
    
    Args:
        keep_score: Giữ nguyên score hiện tại (default: True)
        keep_dynamite: Giữ nguyên số dynamite (default: True)
    """
    global score, goal, pause, start_time, dynamite_count, game_speed, scaled_time_offset, use_fixed_timestep
    
    # Lưu lại giá trị cần giữ
    old_score = score if keep_score else 0
    old_dynamite = dynamite_count if keep_dynamite else 0
    
    # Reset các giá trị khác
    pause = False
    start_time = None
    scaled_time_offset = 0
    
    # Restore giá trị đã lưu
    score = old_score
    dynamite_count = old_dynamite

def load_level_data():
    try:
        # Thử load từ assets/levels/ trước
        try:
            with open('./assets/levels/levels.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback về thư mục gốc
            with open('./levels.json', 'r', encoding='utf-8') as f:
                return json.load(f)
    except FileNotFoundError:
        print("⚠️ Levels.json not found, using empty data")
        return {}
    except Exception as e:
        print(f"❌ Error loading levels.json: {e}")
        return {}

# Load level data khi khởi động
data = load_level_data()
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
scaled_time_offset = 0  # Offset thời gian đã scale theo game speed

def get_time():
    return start_time

def set_time(new_time):
    global start_time
    start_time = new_time

def get_scaled_time_offset():
    """Lấy offset thời gian đã được scale theo game speed"""
    return scaled_time_offset

def add_scaled_time(dt):
    """Thêm delta time đã scale vào offset (được gọi mỗi frame)"""
    global scaled_time_offset
    scaled_time_offset += dt

def reset_scaled_time():
    """Reset scaled time khi bắt đầu level mới"""
    global scaled_time_offset
    scaled_time_offset = 0

current_level = 1
def get_level():
    return current_level
def set_level(new_level):
    global current_level
    current_level = new_level

# Dynamite count management
MAX_DYNAMITE = 5  # Giới hạn tối đa dynamite

def get_dynamite_count():
    return dynamite_count

def set_dynamite_count(count):
    global dynamite_count
    # Giới hạn tối đa 5 dynamite
    if count > MAX_DYNAMITE:
        dynamite_count = MAX_DYNAMITE
    else:
        dynamite_count = max(0, count)  # Không cho phép âm
    return dynamite_count

def add_dynamite(amount=1):
    global dynamite_count
    old_count = dynamite_count
    dynamite_count = min(dynamite_count + amount, MAX_DYNAMITE)  # Giới hạn tối đa
    
    if dynamite_count >= MAX_DYNAMITE:
        print(f"⚠️ Dynamite đã đạt giới hạn tối đa ({MAX_DYNAMITE})!")
    
    return dynamite_count

def use_dynamite():
    global dynamite_count
    if dynamite_count > 0:
        dynamite_count -= 1
        return True
    return False

# AI Action Info (for display)
ai_action_info = {
    'action': None,
    'q_value': None,
    'used_model': False,
    'mode': 'model'  # 'model', 'random', 'selective_random'
}

def get_ai_action_info():
    return ai_action_info

def set_ai_action_info(action, q_value, used_model, mode='model'):
    """
    Lưu thông tin action để hiển thị.
    
    Args:
        action: Action index
        q_value: Q-value của action
        used_model: True nếu dùng model
        mode: 'model' (greedy), 'random' (epsilon/warmup), 'selective_random' (sau khi miss)
    """
    global ai_action_info
    ai_action_info = {
        'action': action,
        'q_value': q_value,
        'used_model': used_model,
        'mode': mode
    }

# Game speed management
SPEED_LEVELS = [1, 2, 5, 10]  # Các mức tốc độ: x1, x2, x5, x10

def get_game_speed():
    return game_speed

def cycle_game_speed():
    """Chuyển đổi tốc độ game theo chu kỳ: 1x -> 2x -> 5x -> 10x -> 1x"""
    global game_speed
    current_index = SPEED_LEVELS.index(game_speed) if game_speed in SPEED_LEVELS else 0
    next_index = (current_index + 1) % len(SPEED_LEVELS)
    game_speed = SPEED_LEVELS[next_index]
    print(f"⚡ Game speed: x{game_speed}")
    return game_speed

def set_game_speed(speed):
    global game_speed
    if speed in SPEED_LEVELS:
        game_speed = speed
    return game_speed

def get_use_fixed_timestep():
    """Lấy flag fixed timestep (True = RL training, False = game bình thường)"""
    return use_fixed_timestep

def set_use_fixed_timestep(value):
    """Set flag fixed timestep cho RL training"""
    global use_fixed_timestep
    use_fixed_timestep = value
    return use_fixed_timestep

# Đường dẫn đến file txt
high_score_file = "high_scores.txt"
high_scores = []