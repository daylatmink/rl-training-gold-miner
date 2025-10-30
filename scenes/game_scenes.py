# -*- coding: utf-8 -*-
from entities.miner import Miner
from entities.rope import Rope
from entities.explosive import Explosive
from entities.button import Button
from entities.shopkeeper import Shopkeeper
from scenes.scene import Scene
from scenes.util import *
clock = pygame.time.Clock()
# THÊM IMPORT
from scenes.util import level_manager, generate_random_level, get_training_level
# THÊM BIẾN TOÀN CỤC ĐỂ CONFIG
USE_GENERATED_LEVELS = True   
class SceneMananger(object):
    def __init__(self):
        self.go_to(StartScene())
    def go_to(self, scene):
        self.scene = scene
        self.scene.manager = self
class StartScene(object):
    def __init__(self):
        super(StartScene, self).__init__()
        self.font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), 48)
        self.button = Button(120, 20, gold_image, 2)
        self.higt_score_btn = Button(80, 500, hight_score, 1)
        self.toggle_level_btn = Button(400, 500, hight_score, 1)
    
    def start(self):
        # 🎯 RESET STATE KHI BẮT ĐẦU GAME MỚI
        from define import reset_game_state
        reset_game_state()
        
        set_time(pygame.time.get_ticks()/1000)
        self.manager.go_to(GameScene(level=get_level()))
    
    def render(self, screen):
        screen.blit(start_BG, (0,0))
        self.button.render(screen)
        self.higt_score_btn.render(screen)
        self.toggle_level_btn.render(screen)  # THÊM
        screen.blit(miner_menu, miner_menu_rect)
        text = self.font.render('Chơi', True, (255, 255, 255))
        screen.blit(text, (250, 160))
        
        # THÊM: Hiển thị trạng thái current level system
        level_type = "Generated Levels" if USE_GENERATED_LEVELS else "Original Levels"
        type_font = pygame.font.Font(None, 24)
        screen.blit(type_font.render(f"Level System: {level_type}", True, (255, 255, 255)), (400, 550))
    def update(self, screen):
        pass
    def handle_events(self, events):
        if self.button.is_click():
            self.start()
        if self.higt_score_btn.is_click():
            self.manager.go_to(HighScoreScene())
        # THÊM: Toggle level system
        if self.toggle_level_btn.is_click():
            global USE_GENERATED_LEVELS
            USE_GENERATED_LEVELS = not USE_GENERATED_LEVELS
class FinishScene(object):
    def __init__(self):
        super(FinishScene, self).__init__()
        self.font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), 28)
        load_sound("goal_sound")
    def render(self, screen):
        screen.blit(cut_scene,(0,0))
        screen.blit(panel_image,panel_image.get_rect(center = (screen_width/2,screen_height/2)))
        screen.blit(text_game_image,text_game_image.get_rect(center = (screen_width/2,200)))
        text = 'Level Up!\nNhấn phím Space để tiếp tục'
        blit_text(screen,text,(377,330),self.font,color=(255,255,255))
    def update(self,screen):
        pass
    def handle_events(self, events):
        for e in events:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                set_time(pygame.time.get_ticks()/1000)
                self.manager.go_to(StoreScene())
class FailureScene(object):
    def __init__(self):
        super(FailureScene, self).__init__()
        write_high_score(get_score())
        load_sound("made_goal_sound")
        self.font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), 24)
    
    def handle_events(self, events):
        for e in events:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                # 🎯 RESET STATE KHI CHƠI LẠI SAU FAILURE
                from define import reset_game_state
                reset_game_state()
                self.manager.go_to(StartScene())
    def render(self, screen):
        screen.blit(cut_scene,(0,0))
        screen.blit(panel_image,panel_image.get_rect(center = (screen_width/2,screen_height/2)))
        screen.blit(text_game_image,text_game_image.get_rect(center = (screen_width/2,200)))
        text = 'Bạn đã không đạt đủ điểm yêu cầu!\nBấm phím Space để chơi lại'
        blit_text(screen,text,(377,350),self.font,color=(255,255,255))
    def update(self,screen):
        pass
    def handle_events(self, events):
        for e in events:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                self.manager.go_to(StartScene())
class WinScene(object):
    def __init__(self):
        super(WinScene, self).__init__()
        write_high_score(get_score())
        load_sound("goal_sound")
        self.font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), 24)
    
    def handle_events(self, events):
        for e in events:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                # 🎯 RESET STATE KHI CHƠI LẠI SAU WIN
                from define import reset_game_state
                reset_game_state()
                self.manager.go_to(StartScene())
    def render(self, screen):
        screen.blit(cut_scene,(0,0))
        screen.blit(panel_image,panel_image.get_rect(center = (screen_width/2,screen_height/2)))
        screen.blit(text_game_image,text_game_image.get_rect(center = (screen_width/2,200)))
        text = 'Chúc mừng bạn đã chiến thắng\ntrong trò chơi này!\n\nNhấn phím Space để chơi lại'
        blit_text(screen,text,(377,300),self.font,color=(255,255,255))
    def update(self,screen):
        pass
    def handle_events(self, events):
        for e in events:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                self.manager.go_to(StartScene())
class HighScoreScene(object):
    def __init__(self):
        super(HighScoreScene, self).__init__()
        self.font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), 24)
        self.continute = Button(1050,50,continue_img,0.5)
    def render(self, screen):
        screen.blit(cut_scene,(0,0))
        screen.blit(panel_image,panel_image.get_rect(center = (screen_width/2,screen_height/2)))
        screen.blit(text_game_image,text_game_image.get_rect(center = (screen_width/2,200)))
        screen.blit(self.font.render('ĐIỂM CAO', True, (255, 255, 255)), (560, 300))
        self.continute.render(screen)
        text = get_high_score_as_text()
        blit_text(screen,text,(377,350),self.font,color=(255,255,255))
    def update(self,screen):
        pass
    def handle_events(self, events):
        if self.continute.is_click():
            self.manager.go_to(StartScene())
                
class StoreScene(object):
    def __init__(self):
        super(StoreScene, self).__init__()
        self.font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), 28)
        self.shopkeeper = Shopkeeper(900,250)
        self.continute = Button(1050,50,continue_img,0.5)

        self.rock_collectors_book = Button(87,420,rock_collectors_book,2)
        self.is_rock = random.randint(0,1)
        if self.is_rock:
            self.rock_price = random.randint(10,150)

        self.strength_drink = Button(300,400,strength_drink,2)
        self.is_strength_drink = random.randint(0,1)
        if self.is_strength_drink:
            self.strength_drink_price = random.randint(0,300)+100

        self.gem_polish = Button(500,440,gem_polish,2)
        self.is_gem_polish = random.randint(0,1)
        if self.is_gem_polish:
            self.gem_polish_price = random.randint(0,get_level()*100) +200

        self.clover = Button(650,420,clover,2)
        self.is_clover = random.randint(0,1)
        if self.is_clover:
            self.clover_price = random.randint(0,get_level()*50) + get_level()*2 + 1

        self.dynamite = Button(800,425,dynamite_shop,2)
        self.is_dynamite = random.randint(0,1)
        if self.is_dynamite:
            self.dynamite_price = random.randint(0,300) + 1 + get_level()*2

        self.text = 'Click vào vật phẩm mà bạn muốn mua\nClick vào tiếp tục khi bạn đã sẵn sàng'
        self.is_buy = False

        self.buyTNT = 0
        self.buyRock = False
        self.buyGem = False
        self.buyClover = False
        self.buyDrink = 1
    def render(self, screen):            
        screen.blit(store_BG,(0,0))
        screen.blit(self.font.render("Tiền: "+str(get_score()), True, (0, 0, 0)), (5, 0))
        self.shopkeeper.draw(screen)
        screen.blit(table_image,table_image.get_rect(bottom = screen_height))
        screen.blit(dialog_image,(220,100))
        self.continute.render(screen)
        if self.is_rock:
            self.rock_collectors_book.render(screen)
            blit_nor_text(screen,"$"+str(self.rock_price),(140,565),self.font,color=(0,150,0))
        if self.is_strength_drink:
            self.strength_drink.render(screen)
            blit_nor_text(screen,"$"+str(self.strength_drink_price),(350,565),self.font,color=(0,150,0))
        if self.is_gem_polish:
            self.gem_polish.render(screen)
            blit_nor_text(screen,"$"+str(self.gem_polish_price),(550,565),self.font,color=(0,150,0))
        if self.is_clover:
            self.clover.render(screen)
            blit_nor_text(screen,"$"+str(self.clover_price),(690,565),self.font,color=(0,150,0))
        if self.is_dynamite:
            self.dynamite.render(screen)
            blit_nor_text(screen,"$"+str(self.dynamite_price),(820,565),self.font,color=(0,150,0))
        blit_text(screen,self.text,(250,110),self.font,color=(0,0,0))
    def update(self,screen): #handel hover
        if self.is_rock:
            if self.rock_collectors_book.is_hover():
                text = "Sách Người sưu tầm đá. Đá sẽ có giá trị\ngấp ba lần ở cấp độ tiếp theo.\nChỉ áp dụng cho 1 cấp độ."
                blit_text(screen,text,(250,620),self.font,color=(255,255,255))
        if self.is_strength_drink:
            if self.strength_drink.is_hover():
                text = "Nước tăng lực. Tốc độ kéo vật phẩm của bạn sẽ\nnhanh hơn một chút ở cấp độ tiếp theo.\nĐồ uống chỉ kéo dài trong một cấp độ."
                blit_text(screen,text,(250,620),self.font,color=(255,255,255))
        if self.is_gem_polish:
            if self.gem_polish.is_hover():
                text = "Đánh bóng đá quý. Trong cấp độ tiếp theo,\nđá quý và kim cương sẽ có giá trị cao hơn.\nChỉ áp dụng cho 1 cấp độ."
                blit_text(screen,text,(250,620),self.font,color=(255,255,255))
        if self.is_clover:
            if self.clover.is_hover():
                text = "Cỏ may mắn. Vật phẩm này sẽ tăng cơ hội\nnhận được thứ gì đó tốt từ túi ở cấp độ tiếp theo.\nChỉ áp dụng cho 1 cấp độ."
                blit_text(screen,text,(250,620),self.font,color=(255,255,255))
        if self.is_dynamite:
            if self.dynamite.is_hover():
                text = "Sau khi bạn kéo được thứ gì đó không có giá trị,\nhãy nhấn phím lên để ném một mảnh thuốc nổ\nvào nó và cho nổ tung nó."
                blit_text(screen,text,(250,620),self.font,color=(255,255,255))
    def handle_events(self, events):
        if self.is_rock:
            if self.rock_collectors_book.is_click():
                if buy_item(1,self.rock_price):
                    self.is_buy = True
                    self.buyRock = True
                    self.is_rock = False
        if self.is_strength_drink:
            if self.strength_drink.is_click():
                if buy_item(2,self.strength_drink_price):
                    self.is_buy = True
                    self.buyDrink = 2
                    self.is_strength_drink = False
        if self.is_gem_polish:
            if self.gem_polish.is_click():
                if buy_item(3,self.gem_polish_price):
                    self.is_buy = True
                    self.buyGem = True
                    self.is_gem_polish = False
        if self.is_clover:
            if self.clover.is_click():
                if buy_item(4,self.clover_price):
                    self.is_buy = True
                    self.buyClover = True
                    self.is_clover = False
        if self.is_dynamite:
            if self.dynamite.is_click():
                if buy_item(5,self.dynamite_price):
                    self.is_buy = True
                    self.buyTNT =1
                    self.is_dynamite = False
        if self.continute.is_click():
            if not(self.is_buy):    
                # self.shopkeeper.current_frame = 1
                # self.render(screen)
                # pygame.time.wait(2000)
                pass #need fix
            self.start(self.buyTNT,self.buyDrink,self.buyClover,self.buyGem,self.buyRock)
        for e in events:
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                self.start(self.buyTNT,self.buyDrink,self.buyClover,self.buyGem,self.buyRock)
    def start(self, tnt=0, speed=1, clover=0, gem=0, rock=0):
        set_time(pygame.time.get_ticks()/1000)
        self.manager.go_to(GameScene(get_level(), tnt, speed, clover, gem, rock, use_generated=USE_GENERATED_LEVELS))
class GameScene(Scene):
    def __init__(self, level, tnt=0, speed=1, is_clover=False, is_gem=False, is_rock=False, use_generated=None):
        super(GameScene, self).__init__()
        self.level = level
        
        # THÊM: Xác định có dùng generated levels không
        if use_generated is None:
            self.use_generated = USE_GENERATED_LEVELS
        else:
            self.use_generated = use_generated
            
        print(f"🎮 GAMESCENE INIT: Level={level}, Use Generated={self.use_generated}")
        
        self.miner = Miner(620, -7, 5)
        self.rope = Rope(643, 45, 300, hoo_images, tnt, speed)
        
        # THAY ĐỔI QUAN TRỌNG: Load level với hệ thống mới
        if self.use_generated:
            # Sử dụng generated levels
            level_id = self._get_generated_level_id()
            print(f"🎮 GAMESCENE: Loading generated level: {level_id}")
            self.bg, self.items = load_level(level_id, is_clover, is_gem, is_rock)
            self.current_level_id = level_id
        else:
            # Sử dụng level cũ từ JSON
            level_id = random_level(self.level)
            print(f"🎮 GAMESCENE: Loading original level: {level_id}")
            self.bg, self.items = load_level(level_id, is_clover, is_gem, is_rock)
            self.current_level_id = level_id
        
        print(f"🎮 GAMESCENE: Loaded {len(self.items)} items")
        
        # self.bg,self.items = load_level("LDEBUG")
        self.play_Explosive = False
        self.explosive = None
        self.text_font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), 14)
        self.timer = 0
        self.pause_time = 0
        self.pause = False
        self.exit_button = Button(1050, 5, exit_image, 0.25)
        self.next_button = Button(950, 0, next_image, 0.4)
    def _get_generated_level_id(self):
        """Tạo hoặc lấy level_id cho generated levels"""
        try:
            # Dựa trên level number để xác định difficulty
            difficulties = ["easy", "medium", "hard", "expert"]
            difficulty_index = min(self.level - 1, len(difficulties) - 1)
            difficulty = difficulties[difficulty_index]
            
            print(f"🎯 DEBUG: Creating generated level for level {self.level} with difficulty {difficulty}")
            
            # 🎯 Tạo level_id unique
            level_id = f"GEN_L{self.level}_{random.randint(1000, 9999)}"
            
            print(f"🎯 DEBUG: Level ID: {level_id}")
            
            # 🎯 Đảm bảo level được tạo trong manager
            level_data = level_manager.get_level(level_id, difficulty)
            
            if level_data and level_data.get('entities'):
                print(f"✅ DEBUG: Successfully generated level {level_id} with {len(level_data['entities'])} entities")
                return level_id
            else:
                print(f"❌ DEBUG: Failed to generate level {level_id}")
                raise Exception("Level generation failed")
            
        except Exception as e:
            print(f"❌ Error in _get_generated_level_id: {e}")
            import traceback
            traceback.print_exc()
            # Fallback về level gốc
            ran_level = random.randint(1, 3)
            fallback_level = f"L{self.level}_{ran_level}"
            print(f"🔄 Fallback to original level: {fallback_level}")
            return fallback_level
    def render_debug_info(self, screen):
        """Hiển thị thông tin debug cho generated levels (tùy chọn)"""
        if self.use_generated:
            debug_font = pygame.font.Font(None, 20)
            debug_text = f"Level: {self.current_level_id} | Entities: {len(self.items)}"
            screen.blit(debug_font.render(debug_text, True, (255, 0, 0)), (10, 50))
            
            # Hiển thị số lượng từng loại entity
            entity_count = {}
            for item in self.items:
                item_type = type(item).__name__
                entity_count[item_type] = entity_count.get(item_type, 0) + 1
            
            y_offset = 70
            for entity_type, count in entity_count.items():
                screen.blit(debug_font.render(f"{entity_type}: {count}", True, (255, 0, 0)), (10, y_offset))
                y_offset += 20
    def render(self, screen):            
        dt = clock.tick(60) / 1000
        
        if self.rope.state == 'retracting' and not(self.rope.is_use_TNT):
            self.miner.state = 2
            
        screen.blit(bg_top,(0,0))
        screen.blit(self.bg,(0,72))
        self.exit_button.render(screen)
        self.next_button.render(screen)
        
        #Draw item - SỬA: items bây giờ là list
        for item in self.items:
            item.draw(dt, screen)

        # DEBUG: Hiển thị thông tin items
        debug_font = pygame.font.Font(None, 24)
        screen.blit(debug_font.render(f"Items: {len(self.items)}", True, (255, 0, 0)), (10, 100))
        screen.blit(debug_font.render(f"Level: {self.current_level_id}", True, (255, 0, 0)), (10, 120))
        screen.blit(debug_font.render(f"Use Generated: {self.use_generated}", True, (255, 0, 0)), (10, 140))

        # SỬA: Kiểm tra explosive tồn tại trước khi truy cập thuộc tính
        if self.play_Explosive and self.explosive is not None:
            load_sound("explosive_sound")
            self.explosive.draw(screen)
            self.explosive.update(dt)
            if hasattr(self.explosive, 'is_exit') and self.explosive.is_exit:
                self.explosive = None
                self.play_Explosive = False
                self.miner.is_TNT = False
                self.miner.state = 0
                self.rope.is_use_TNT = False
                
        for i in range(self.rope.have_TNT):
            screen.blit(dynamite_image, (725+i*25, 10))
        
        #Update sprite
        self.miner.update(dt)
        self.miner.draw(screen)
        self.rope.update(self.miner, dt, screen)
        self.rope.draw(screen)
        draw_point(self.rope, dt, self.miner)

    def update(self, screen):
        self.timer = 60 - int(pygame.time.get_ticks()/1000 - get_time())
        screen.blit(self.text_font.render("Tiền:", True, (0, 0, 0)), (5, 0))
        screen.blit(self.text_font.render("$"+str(get_score()), True, (0, 150, 0)), (55, 0))
        screen.blit(self.text_font.render("Mục tiêu:", True, (0, 0, 0)), (5, 25))
        screen.blit(self.text_font.render("$"+str(get_goal()), True, (255, 150, 0)), (96, 25))
        screen.blit(self.text_font.render("Thời gian:", True, (0, 0, 0)), (1140, 0))
        screen.blit(self.text_font.render(str(self.timer), True, (255, 100, 7)), (1240, 0))
        screen.blit(self.text_font.render("Cấp:", True, (0, 0, 0)), (1140, 25))
        screen.blit(self.text_font.render(str(self.level), True, (255, 100, 7)), (1190, 25))
        
        # THÊM: Xử lý collision trong update
        if self.miner.state == 1:
            items_to_remove = []
            for item in self.items:
                if is_collision(self.rope, item):
                    self.rope.item = item
                    if hasattr(item, 'is_move'):
                        item.is_move = False
                    if hasattr(item, 'is_explosive') and item.is_explosive == True:
                        load_sound("explosive_sound")
                        explosive_item(item, self.items)
                    self.rope.state = 'retracting'
                    items_to_remove.append(item)
                    break  # Chỉ bắt 1 item tại một thời điểm
            
            # Xóa items sau khi đã lặp xong
            for item in items_to_remove:
                self.items.remove(item)
            self.timer = 60 - int(pygame.time.get_ticks()/1000 - get_time())
            screen.blit(self.text_font.render("Tiền:", True, (0, 0, 0)), (5, 0))
            screen.blit(self.text_font.render("$"+str(get_score()), True, (0, 150, 0)), (55, 0))
            screen.blit(self.text_font.render("Mục tiêu:", True, (0, 0, 0)), (5, 25))
            screen.blit(self.text_font.render("$"+str(get_goal()), True, (255, 150, 0)), (96, 25))
            screen.blit(self.text_font.render("Thời gian:", True, (0, 0, 0)), (1140, 0))
            screen.blit(self.text_font.render(str(self.timer), True, (255, 100, 7)), (1240, 0))
            screen.blit(self.text_font.render("Cấp:", True, (0, 0, 0)), (1140, 25))
            screen.blit(self.text_font.render(str(self.level), True, (255, 100, 7)), (1190, 25))
    def next_level(self):
        if get_score() > get_goal():
            set_level(get_level() + 1)
            
            # if get_level() > 10:
            #     self.manager.go_to(WinScene())
            #     return
                
            set_goal(get_goal() + get_level() * goalAddOn)
            self.manager.go_to(FinishScene())
        else:
            set_level(1)
            set_goal(650)
            self.manager.go_to(FailureScene())
    def handle_events(self, events):
        if self.timer < 0:
            self.next_level()
            
        for e in events:
            if self.exit_button.is_click():
                self.manager.go_to(StartScene())
            if self.next_button.is_click():
                self.next_level()
            if e.type == pygame.QUIT:
                write_high_score(get_score())
                pygame.quit()
                sys.exit(0)
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: #PAUSE and UNPAUSE
                    self.pause = not self.pause
                    if self.pause: #PAUSE
                        self.pause_time = pygame.time.get_ticks()/1000
                        set_pause(True)
                    else: #UNPAUSE
                        set_pause(False)
                        set_time(get_time() + pygame.time.get_ticks()/1000 - self.pause_time)
                if e.key == pygame.K_ESCAPE: #ESC --> test
                    self.next_level()
                if e.key == pygame.K_DOWN and self.rope.timer <= 0: # expanding
                    self.miner.state = 1
                if e.key == pygame.K_UP: # retracting
                    if hasattr(self.rope, 'have_TNT') and self.rope.have_TNT > 0 and self.rope.item is not None:
                        self.rope.is_use_TNT = True
                        self.miner.state = 4
                        # SỬA: Khởi tạo explosive đúng cách
                        self.explosive = Explosive(self.rope.x2-128, self.rope.y2-128, 12)
                        self.play_Explosive = True
                        self.rope.have_TNT -= 1
                        self.rope.length = 50
                        self.miner.is_TNT = True
    def update(self, screen):
        self.timer = 60 - int(pygame.time.get_ticks()/1000 - get_time())
        screen.blit(self.text_font.render("Tiền:", True, (0, 0, 0)), (5, 0))
        screen.blit(self.text_font.render("$"+str(get_score()), True, (0, 150, 0)), (55, 0))
        screen.blit(self.text_font.render("Mục tiêu:", True, (0, 0, 0)), (5, 25))
        screen.blit(self.text_font.render("$"+str(get_goal()), True, (255, 150, 0)), (96, 25))
        screen.blit(self.text_font.render("Thời gian:", True, (0, 0, 0)), (1140, 0))
        screen.blit(self.text_font.render(str(self.timer), True, (255, 100, 7)), (1240, 0))
        screen.blit(self.text_font.render("Cấp:", True, (0, 0, 0)), (1140, 25))
        screen.blit(self.text_font.render(str(self.level), True, (255, 100, 7)), (1190, 25))
    
        # THÊM: Xử lý collision trong update thay vì render
        if self.miner.state == 1:
            items_to_remove = []
            for item in self.items:
                if is_collision(self.rope, item):
                    self.rope.item = item
                    self.rope.item.is_move = False
                    if hasattr(item, 'is_explosive') and item.is_explosive == True:
                        load_sound("explosive_sound")
                        explosive_item(item, self.items)
                    self.rope.state = 'retracting'
                    items_to_remove.append(item)
                    break  # Chỉ bắt 1 item tại một thời điểm
        
            # Xóa items sau khi đã lặp xong
            for item in items_to_remove:
                self.items.remove(item)
class RLTrainingScene(Scene):
    """Scene đặc biệt cho RL training với generated levels"""
    def __init__(self, episode=None, difficulty=None):
        super(RLTrainingScene, self).__init__()
        
        # Lấy level cho training
        if difficulty:
            level_data = level_manager.get_level(f"TRAIN_{episode or random.randint(1, 10000)}", difficulty)
        else:
            level_data = get_training_level(episode)
            
        self.level_data = level_data
        self.miner = Miner(620, -7, 5)
        self.rope = Rope(643, 45, 300, hoo_images, 0, 1)
        
        # Load items từ level_data
        self.bg, self.items = load_level(f"TRAIN_{episode or 0}", False, False, False)
        
        self.text_font = pygame.font.Font(os.path.join("assets", "fonts", 'Fernando.ttf'), 14)
        self.timer = 0
        self.episode = episode
        
    def render(self, screen):
        # Tương tự như GameScene nhưng không có UI điều khiển
        dt = clock.tick(60) / 1000
        # ... rendering logic giống GameScene ...
        
    def get_state(self):
        """Trả về state cho RL agent"""
        # Ví dụ: trả về thông tin về các items, vị trí, etc.
        state = {
            'miner_pos': (self.miner.x, self.miner.y),
            'rope_angle': self.rope.angle,
            'rope_length': self.rope.length,
            'items': [(type(item).__name__, item.x, item.y) for item in self.items],
            'score': get_score(),
            'time_left': self.timer
        }
        return state
        
    def apply_action(self, action):
        """Áp dụng action từ RL agent"""
        # action có thể là: 0=không làm gì, 1=thả dây, 2=kéo dây, 3=ném TNT, etc.
        if action == 1 and self.rope.state == 'idle':
            self.rope.state = 'expanding'
        elif action == 2 and self.rope.state == 'expanding':
            self.rope.state = 'retracting'
        # ... các action khác ...   
    def update(seft,screen):
        pass