import pygame
import sys
from scenes.game_scenes import SceneMananger
from define import reset_game_state 

def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 820))
    pygame.display.set_caption("Gold Miner")
    reset_game_state()
    
    manager = SceneMananger()
    
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        manager.scene.handle_events(events)
        manager.scene.render(screen)
        manager.scene.update(screen)
        pygame.display.update()

if __name__ == "__main__":
    main()