# Copyright (c) CUBOX, Inc. and its affiliates.
import pygame
from pygame.locals import *

class Button:
    """Button class for Pygame UI"""
    def __init__(self, rect, text, font, callback=None, is_toggle=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.is_toggle = is_toggle
        self.is_active = False
        
    def draw(self, surface):
        # Draw button background
        color = (100, 100, 255) if self.is_active else (50, 50, 50)
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 1)  # Border
        
        # Draw text centered
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def handle_event(self, event, state):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.is_toggle:
                    self.is_active = not self.is_active
                if self.callback:
                    self.callback(state)
                return True
        return False