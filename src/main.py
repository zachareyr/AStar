from gc import callbacks
import sys
import pygame
import numpy as np
from enum import Enum
from typing import Tuple, List
import math

# Return the euclidean distance between start and end
# start: int[2], end: int[2]


def euclidean_distance(start: List[int], end: List[int]) -> float:
    return math.sqrt((start[0]-end[0])*(start[0]-end[0]) + (start[1]-end[1])*(start[1]-end[1]))

# Vertical distance + Horizontal distance


# A cell for use in the A* pathfinding algorithm
class Cell:
    def __init__(self, position: Tuple[int, int], goal: Tuple[int, int], distance_callback, parent=None) -> None:
        self.parent = parent
        self.position: Tuple[int, int] = position
        self.goal: Tuple[int, int] = goal
        self.cost: int = 0  # AKA G
        if parent is None:
            self.cost = 1
        else:
            self.cost = parent.cost + 1

        self.estimated_cost_to_end: float = distance_callback(
            position, goal)  # AKA H

        self.total_cost: float = self.cost + self.estimated_cost_to_end  # AKA F

    def __eq__(self, other) -> bool:
        return self.position == other.position

# Represents colors in a way readable to pygame


class Color(Enum):
    WHITE = pygame.Color(255, 255, 255),
    BLACK = pygame.Color(0, 0, 0),
    LIGHT_BLUE = pygame.Color(27, 200, 247),
    RED = pygame.Color(204, 0, 0),
    GREEN = pygame.Color(29, 156, 9),


# Links board states to their corresponding colors
class BOARD_STATE_COLORS(Enum):
    OFF = Color.WHITE.value[0],        # Default for board, no activity
    ON = Color.BLACK.value[0],         # Has a block on it
    START = Color.RED.value[0],        # The selected starting point
    END = Color.GREEN.value[0],        # The selected ending point
    PATH = Color.LIGHT_BLUE.value[0],  # Path generated between start and end


# The states to be represented in the numpy array 'board'
class BOARD_STATES(Enum):
    OFF: int = 0,    # Default for board, no activity
    ON: int = 1,     # Has a block on it
    START: int = 2,  # The selected starting point
    END: int = 3,    # The selected ending point
    PATH: int = 4,   # Path generated between start and end


# A simple class to render a rect on a surface
class RenderedObject:
    renderable_objects = []

    def __init__(self, surface, color, rect: Tuple[int, int, int, int], active: bool = True) -> None:
        self.surface = surface
        self.color = color
        self.rect: Tuple[int, int, int, int] = rect
        self.active: bool = active

        self.left: int = rect[0]
        self.top: int = rect[1]
        self.width: int = rect[2]
        self.height: int = rect[3]

        self.rendered_surface = pygame.Surface((self.width, self.height))
        self.rendered_surface.fill(self.color)

        # A static List to track all instances of rendered object. This will be used to determine what to render in the loop
        RenderedObject.renderable_objects.append(self)

    # Blits the object on top of the screen
    def render(self) -> None:
        self.rendered_surface.fill(self.color)
        self.surface.blit(self.rendered_surface, (self.left, self.top))


# A render-able object with collision detection
class CollidableObject(RenderedObject):
    def __init__(self, surface, color, rect: Tuple[int, int, int, int], active=True) -> None:
        # Note that RenderedObject.__init__(..args) must be used instead of super().__init__(..args) because of diamond inheritance
        RenderedObject.__init__(self, surface, color, rect, active)

    # Returns whether or not the object is colliding with a given position
    def is_colliding(self, position) -> bool:
        return position[0] >= self.left and position[0] <= self.left+self.width and position[1] >= self.top and position[1] <= self.top+self.height

    def render(self) -> None:
        RenderedObject.render(self)


# An object with a square background, capable of rendering text
class TextObject(RenderedObject):
    def __init__(self, surface, color, rect, text: str, font, padding: Tuple[int, int] = (0, 0), centered: Tuple[bool, bool] = (False, False), active=True) -> None:
        RenderedObject.__init__(self, surface, color, rect, active)
        self.text: str = text
        self.font = font
        self.render_surface: pygame.Surface = self.font.render(
            text, False, Color.BLACK.value[0])
        self.text_size: Tuple[int, int] = self.render_surface.get_size()
        self.left_padding: int = padding[0]
        self.top_padding: int = padding[1]
        self.centered_horzontal: bool = centered[0]
        self.centered_vertical: bool = centered[1]

        if self.centered_horzontal and self.left_padding != 0:
            raise Exception("Text cannot be centered with padding.")

        if self.centered_vertical and self.top_padding != 0:
            raise Exception("Text cannot be centered with padding.")

    # Renders both the rect of the object and the text on top of it
    def render(self) -> None:
        RenderedObject.render(self)
        text_x_position: int = self.left
        # Center the text if it should be centered, otherwise add the specified padding to it. Repeat for X and Y
        if self.centered_horzontal:
            text_x_position += self.width // 2 - self.text_size[0] // 2
        else:
            text_x_position += self.left_padding

        text_y_position: int = self.top
        if self.centered_vertical:
            text_y_position += self.height // 2 - self.text_size[1] // 2
        else:
            text_y_position += self.top_padding

        # Put the text on the screen
        screen.blit(self.render_surface, (text_x_position, text_y_position))


# A renderable, clickable object with text on it
class Button(CollidableObject, TextObject):
    buttons = []

    def __init__(self, surface, color: pygame.Color, rect: pygame.Rect, text: str, font, callback, padding: Tuple[int, int] = (0, 0), centered: Tuple[bool, bool] = (False, False), active=True) -> None:
        CollidableObject.__init__(self, surface, color, rect, active)
        TextObject.__init__(self, surface, color, rect,
                            text, font, padding, centered, active)
        # The function to be called when the button is pressed
        self.callback = callback
        Button.buttons.append(self)

    def is_colliding(self, position) -> None:
        return CollidableObject.is_colliding(self, position)

    def is_clicked(self, event) -> bool:
        return self.is_colliding(pygame.mouse.get_pos()) and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1

    def click(self) -> None:
        self.callback()

    def render(self) -> None:
        TextObject.render(self)


# A class to represent the grid of squares
class Board:
    def __init__(self, width: int, height: int) -> None:
        # The position of the start square or None if there isn't one
        self.start: Tuple[int, int] = None
        # The position of the end square or None if there isn't one
        self.end: Tuple[int, int] = None
        # Whether or not the user is currently placing the start square
        self.placing_start: bool = False
        # Whether or not the user is currently placing the end square
        self.placing_end: bool = False
        self.width: int = width
        self.height: int = height
        self.path_drawn = False
        # For the meanings of positions on the board, see the BOARD_STATES enum [line 28]
        self.board: np.ndarray = np.zeros((self.height, self.width), dtype=int)
        self.diagonal = True

    # Sets the value at a position in the board, according to a !!!PIXEL VALUE!!! (NOT AN ARRAY INDEX)
    def draw_position(self, position: Tuple[int, int], value: int) -> int:

        # Automatically remove any drawn paths when the user attempts to draw again

        x: int = position[0] // PIXELS_PER_UNIT
        y: int = position[1] // PIXELS_PER_UNIT

        # If the user clicks out of bounds, exit early
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 1

        if self.path_drawn:
            self.remove_paths(message=False)

        # If there is already something drawn at the position, don't draw there
        if self.board[y, x] != BOARD_STATES.OFF.value[0]:
            return 1

        # If the start or end is being placed, mark their positions
        if value == BOARD_STATES.START.value[0]:
            self.start = (y, x)
        elif value == BOARD_STATES.END.value[0]:
            self.end = (y, x)

        # Set the board at [y,x] to value
        self.board[y, x] = value
        return 0

    # Replaces the value at the position with an empty space
    def erase(self, position: Tuple[int, int]) -> int:
        # Automatically remove any drawn paths when the user attempts to draw again
        if self.path_drawn:
            self.remove_paths(message=False)

        x: int = position[0] // PIXELS_PER_UNIT
        y: int = position[1] // PIXELS_PER_UNIT

        # If the area is out of bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 1

        # If the start or end is being removed, mark that they no longer exist
        if self.board[y, x] == BOARD_STATES.START.value[0]:
            self.start = None
        elif self.board[y, x] == BOARD_STATES.END.value[0]:
            self.end = None

        # If all of the previous tests are passed, set the position to empty
        self.board[y, x] = BOARD_STATES.OFF.value[0]
        return 0

    # Draws the lines separating the cells
    def draw_lines(self, surface, color):
        for i in range(self.width):
            pygame.draw.line(surface, color, (i*PIXELS_PER_UNIT, 0),
                             (i*PIXELS_PER_UNIT, self.height*PIXELS_PER_UNIT))

        for i in range(self.height):
            pygame.draw.line(surface, color, (0, i*PIXELS_PER_UNIT),
                             (self.width*PIXELS_PER_UNIT, i*PIXELS_PER_UNIT))

    # Fills in the cells on the pygame display according to the board.board array
    def update_display(self, surface):
        # For every square in the board, draw a rect based on the given surface based on its value
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i, j] == BOARD_STATES.ON.value[0]:
                    pygame.draw.rect(surface, BOARD_STATE_COLORS.ON.value[0], (
                        j*PIXELS_PER_UNIT, i*PIXELS_PER_UNIT, PIXELS_PER_UNIT, PIXELS_PER_UNIT))
                elif self.board[i, j] == BOARD_STATES.OFF.value[0]:
                    pygame.draw.rect(surface, BOARD_STATE_COLORS.OFF.value[0], (
                        j*PIXELS_PER_UNIT, i*PIXELS_PER_UNIT, PIXELS_PER_UNIT, PIXELS_PER_UNIT))
                elif self.board[i, j] == BOARD_STATES.START.value[0]:
                    pygame.draw.rect(surface, BOARD_STATE_COLORS.START.value[0], (
                        j*PIXELS_PER_UNIT, i*PIXELS_PER_UNIT, PIXELS_PER_UNIT, PIXELS_PER_UNIT))
                elif self.board[i, j] == BOARD_STATES.END.value[0]:
                    pygame.draw.rect(surface, BOARD_STATE_COLORS.END.value[0], (
                        j*PIXELS_PER_UNIT, i*PIXELS_PER_UNIT, PIXELS_PER_UNIT, PIXELS_PER_UNIT))
                elif self.board[i, j] == BOARD_STATES.PATH.value[0]:
                    pygame.draw.rect(surface, BOARD_STATE_COLORS.PATH.value[0], (
                        j*PIXELS_PER_UNIT, i*PIXELS_PER_UNIT, PIXELS_PER_UNIT, PIXELS_PER_UNIT))

    # Removes all paths from the board
    def remove_paths(self, message=True):
        if message:
            print("Removing paths")
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i, j] == BOARD_STATES.PATH.value[0]:
                    self.board[i, j] = BOARD_STATES.OFF.value[0]
        self.path_drawn = False

    # Runs A* pathfinding and adds the path to the board. If no path is found, does nothing
    # Returns 0 if successful, 1 otherwise
    def run_pathfinding(self):
        # If there is no start or no end, don't go any further
        if (self.start == None or self.end == None):
            print(f"Failed to run pathfinding. No {0} location",
                  self.start if self.start is None else self.end)
            return 1
        if (self.path_drawn):
            self.remove_paths(message=False)

        print("Running pathfinding")

        # An 'open array' for cells that are open to being added
        open = [Cell(self.start, self.end, euclidean_distance)]
        # A closed array for cells that cannot be added
        closed = []

        # While there are still cells to check
        while len(open) != 0:
            # Set the current node to the node with the lowest predicted cost
            current_node = min(open, key=lambda x: x.total_cost)

            open.remove(current_node)
            closed.append(current_node)

            # If the current node is the end:
            # we're done!
            if current_node.position == self.end:
                # Backtrace via current_path_item.parent, and edit the board, drawing the path
                current_path_item = current_node
                while current_path_item is not None:
                    # Prevents the path from overriding the start and end nodes
                    if self.board[current_path_item.position[0], current_path_item.position[1]] == BOARD_STATES.OFF.value[0]:
                        # Set the board at [current position] to the path
                        self.board[current_path_item.position[0],
                                   current_path_item.position[1]] = BOARD_STATES.PATH.value[0]
                    # Recurse back
                    current_path_item = current_path_item.parent
                print("Done")
                self.path_drawn = True
                return 0

            # Check all adjacent spaces (depending on if diagonal spaces are enabled or not)
            children = []
            for offset in ([(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)] if self.diagonal else [(0, -1), (0, 1), (-1, 0), (1, 0)]):
                # Calculate the position of the 'child'
                child_position = (
                    current_node.position[0] + offset[0], current_node.position[1] + offset[1])
                # If the child has an invalid position, skip to the next one
                if child_position[0] >= self.height or child_position[1] >= self.width or child_position[0] < 0 or child_position[1] < 0:
                    continue

                # If the child is a wall, skip to the next one
                if self.board[child_position[0], child_position[1]] == BOARD_STATES.ON.value[0]:
                    continue

                # Otherwise, add the child to the children array
                # Note that creating a cell automatically calculates its cost according to the callback [euclidean distance]
                children.append(Cell(
                    child_position, self.end, euclidean_distance, current_node))

            # For every !valid! child
            for child in children:
                # Check if the child is in the closed list, and if it is, skip it
                for closed_list_member in closed:
                    if child == closed_list_member:
                        break
                else:
                    # If a better or equal-weighted version of the child is already in the open array, skip it
                    for open_cell in open:
                        if child == open_cell and child.total_cost >= open_cell.total_cost:
                            break
                    else:
                        # If all of the previous checks pass, add it to open
                        open.append(child)

        # If the while loop ends, that means that no valid path was found
        print("Failed to draw path. No path found")

    # Resets the board

    def reset_board(self):
        print("Resetting board")
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.start = None
        self.end = None
        self.placing_end = False
        self.placing_start = False
        self.path_drawn = False

    def place_start(self):
        # Don't place it if there is already a start
        if self.start != None:
            print("Start already placed")
            return
        print("Placing start")
        # Make sure that the user cannot place start and end simultaneously
        self.placing_end = False
        self.placing_start = True

    def place_end(self):

        # Don't place it if there is already an end
        if self.end != None:
            print("End already placed")
            return
        print("Placing end")
        self.placing_end = True
        # Make sure that the user cannot place start and end simultaneously
        self.placing_start = False

    # Returns the current state of the board
    def get_state(self):
        if self.placing_start:
            return BOARD_STATES.START.value[0]
        if self.placing_end:
            return BOARD_STATES.END.value[0]
        return BOARD_STATES.ON.value[0]

    def toggle_diagonal_movement(self):
        self.diagonal = not self.diagonal
        print("Diagonal movement {0}".format(
            "on" if self.diagonal else "off"))


# Setup
pygame.init()
pygame.font.init()

# Globals
PIXELS_PER_UNIT: int = 20
SIZE = WIDTH, HEIGHT = 1200, 1000
SIDEBAR_AREA = 300
DRAWABLE_AREA = DRAWABLE_WIDTH, DRAWABLE_HEIGHT = WIDTH - SIDEBAR_AREA, HEIGHT
screen: pygame.Surface = pygame.display.set_mode(SIZE)
pygame.display.set_caption("A* Pathfinder")
board: Board = Board(DRAWABLE_WIDTH // PIXELS_PER_UNIT,
                     DRAWABLE_HEIGHT // PIXELS_PER_UNIT)
DEFAULT_FONT = pygame.font.SysFont("arial", 24)


def main():
    running: bool = True

    # Define sprites and game objects here, !outside the loop!
    title: TextObject = TextObject(screen, Color.LIGHT_BLUE.value[0], (
        925, 25, 250, 100), "A* Pathfinder", DEFAULT_FONT, centered=(True, True))

    run_button: Button = Button(
        screen, Color.GREEN.value[0], (935, 160, 230, 80), "Start", DEFAULT_FONT, board.run_pathfinding, centered=(True, True))

    reset_button: Button = Button(
        screen, Color.GREEN.value[0], (935, 260, 230, 80), "Reset", DEFAULT_FONT, board.reset_board, centered=(True, True))

    place_start_button: Button = Button(
        screen, Color.GREEN.value[0], (935, 360, 230, 80), "Place Start", DEFAULT_FONT, board.place_start, centered=(True, True))

    place_end_button: Button = Button(
        screen, Color.GREEN.value[0], (935, 460, 230, 80), "Place End", DEFAULT_FONT, board.place_end, centered=(True, True))

    remove_path_button: Button = Button(
        screen, Color.GREEN.value[0], (935, 560, 230, 80), "Remove Paths", DEFAULT_FONT, board.remove_paths, centered=(True, True))

    diagonal_movement_toggle: Button = Button(
        screen, Color.GREEN.value[0], (935, 660, 230, 80), "Diagonal Movement", DEFAULT_FONT, board.toggle_diagonal_movement, centered=(True, True))

    # Game Loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # If left click is pressed
            if pygame.mouse.get_pressed(3)[0]:
                # Set the board to the current state of the board at the mouse pos
                mouse_position = pygame.mouse.get_pos()
                draw_success = board.draw_position(mouse_position,
                                                   board.get_state())
                # If the position is drawn succesfully
                if draw_success == 0:
                    board.placing_start, board.placing_end = False, False
            # If right click is pressed
            if pygame.mouse.get_pressed(3)[2]:
                # Turn the board off at the current mouse pos
                board.erase(pygame.mouse.get_pos())

            # For every button, check if it was clicked, and if it was, execute its callback
            for button in Button.buttons:
                if button.is_clicked(event):
                    button.click()

        # Fill the background in
        screen.fill(Color.LIGHT_BLUE.value[0])

        # Draw the sidebar
        pygame.draw.rect(
            screen, Color.BLACK.value[0], (DRAWABLE_WIDTH, 0, SIDEBAR_AREA, HEIGHT))

        # Update the cells
        board.update_display(screen)

        # Draw the cell-separating lines
        board.draw_lines(screen, Color.BLACK.value[0])

        if board.diagonal:
            diagonal_movement_toggle.color = Color.GREEN.value[0]
        else:
            diagonal_movement_toggle.color = Color.RED.value[0]
        # Render all objects
        for renderable_object in RenderedObject.renderable_objects:
            renderable_object.render()

        # If the user is setting the starting position, tint the screen
        if board.placing_start:
            tmp = pygame.Surface(SIZE)
            tmp.fill(BOARD_STATE_COLORS.START.value[0])
            tmp.set_alpha(100)
            screen.blit(tmp, (0, 0))

        # If the user is placing start, tint the screen
        if board.placing_end:
            tmp = pygame.Surface(SIZE)
            tmp.fill(BOARD_STATE_COLORS.END.value[0])
            tmp.set_alpha(100)
            screen.blit(tmp, (0, 0))

        pygame.display.update()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
