"""
Tiny terminal endless runner to pass the time while training runs.
Tap SPACE to jump over incoming obstacles; stay on the ground otherwise.
Quit with Q.
"""

import curses
import random
import time

PLAYER = "@"
OBSTACLE = "#"
EMPTY = " "

TICK = 0.08
SPAWN_CHANCE = 0.12
JUMP_STRENGTH = -3
GRAVITY = 1


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def draw(stdscr, grid, score, jump_prompt: str) -> None:
    stdscr.erase()
    for y, row in enumerate(grid):
        stdscr.addstr(y, 0, "".join(row))
    stdscr.addstr(len(grid), 0, f"Score: {score}  {jump_prompt}  (Q quits)")
    stdscr.refresh()


def run_game(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    height, width = stdscr.getmaxyx()
    height = max(10, height - 2)
    width = max(20, width - 2)

    ground_y = height - 2
    player_x = width // 6
    player_y = ground_y
    velocity_y = 0
    obstacles: list[tuple[int, int]] = []  # (y, x)
    score = 0
    jump_prompt = "Press SPACE to jump"
    game_over = False

    while not game_over:
        key = stdscr.getch()
        if key != -1:
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" ") and player_y >= ground_y:
                velocity_y = JUMP_STRENGTH

        if random.random() < SPAWN_CHANCE:
            obstacles.append((ground_y, width - 1))

        # Apply gravity and update position.
        velocity_y = clamp(velocity_y + GRAVITY, JUMP_STRENGTH, 4)
        player_y = clamp(player_y + velocity_y, 0, ground_y)

        # Move obstacles left; earn points when safely passed.
        new_obstacles = []
        for oy, ox in obstacles:
            nx = ox - 1
            if nx == player_x and oy == player_y:
                game_over = True
            if nx >= 0:
                new_obstacles.append((oy, nx))
            else:
                score += 1
        obstacles = new_obstacles

        grid = [[EMPTY for _ in range(width)] for _ in range(height)]
        for oy, ox in obstacles:
            grid[oy][ox] = OBSTACLE
        grid[ground_y] = ["_" for _ in range(width)]
        grid[player_y][player_x] = PLAYER

        draw(stdscr, grid, score, jump_prompt)
        time.sleep(TICK)

    stdscr.nodelay(False)
    stdscr.addstr(height // 2, max(0, width // 2 - 5), "Game over. Press any key.")
    stdscr.getch()


def main() -> None:
    curses.wrapper(run_game)


if __name__ == "__main__":
    main()
