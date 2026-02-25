import time

import pygame


class Gamepad(object):
    def __init__(self, index=0):
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(index)
        self.joystick.init()

    def read(self):
        """获取左右摇杆的输入值"""
        pygame.event.pump()  # 没有pump的话, 可能读出来的全是0
        left_stick_x = self.joystick.get_axis(0)
        left_stick_y = self.joystick.get_axis(1)
        right_stick_x = self.joystick.get_axis(2)
        right_stick_y = self.joystick.get_axis(3)
        return left_stick_x, left_stick_y, right_stick_x, right_stick_y

    def test(self):
        screen = pygame.display.set_mode((800, 600))
        font = pygame.font.Font(None, 36)
        pygame.display.set_caption("Gamepad Input")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button {event.button} pressed")
            left_stick_x, left_stick_y, right_stick_x, right_stick_y = self.read()
            left_stick_text = font.render(f"Left Stick: ({left_stick_x:.2f}, {left_stick_y:.2f})",
                                          True, (255, 255, 255))
            right_stick_text = font.render(f"Right Stick: ({right_stick_x:.2f}, {right_stick_y:.2f})",
                                           True, (255, 255, 255))
            screen.fill((0, 0, 0))
            screen.blit(left_stick_text, (10, 10))
            screen.blit(right_stick_text, (10, 50))
            pygame.display.flip()

        pygame.quit()


def main():
    gamepad = Gamepad()
    for i in range(10):
        print(f'## {i}:')
        print(gamepad.read())
        time.sleep(0.7)
    gamepad.test()


if __name__ == "__main__":
    main()
