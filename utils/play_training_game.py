"""
Tiny terminal game to pass the time while training runs.
Move with arrow keys or WASD; press SPACE to shoot falling blocks.
Quit with Q.
"""

import curses
import random
import time


PLAYER = "@"
BLOCK = "#"
BULLET = "|"
EMPTY = " "

TICK = 0.08
SPAWN_CHANCE = 0.15


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def draw(stdscr, grid, score, lives) -> None:
    stdscr.erase()
    for y, row in enumerate(grid):
        stdscr.addstr(y, 0, "".join(row))
    stdscr.addstr(len(grid), 0, f"Score: {score}  Lives: {lives}  (Q quits)")
    stdscr.refresh()


def run_game(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    height, width = stdscr.getmaxyx()
    height = max(10, height - 2)
    width = max(20, width - 2)

    player_x = width // 2
    player_y = height - 1
    blocks = []
    bullets = []
    score = 0
    lives = 3

    def spawn_block():
        x = random.randint(0, width - 1)
        blocks.append([0, x])

    while lives > 0:
        key = stdscr.getch()
        if key != -1:
            if key in (ord("q"), ord("Q")):
                break
            if key in (curses.KEY_LEFT, ord("a")):
                player_x = clamp(player_x - 1, 0, width - 1)
            if key in (curses.KEY_RIGHT, ord("d")):
                player_x = clamp(player_x + 1, 0, width - 1)
            if key in (curses.KEY_UP, ord("w")):
                player_y = clamp(player_y - 1, 0, height - 1)
            if key in (curses.KEY_DOWN, ord("s")):
                player_y = clamp(player_y + 1, 0, height - 1)
            if key == ord(" "):
                bullets.append([player_y - 1, player_x])

        if random.random() < SPAWN_CHANCE:
            spawn_block()

        # Move bullets up.
        new_bullets = []
        for y, x in bullets:
            ny = y - 1
            if ny >= 0:
                new_bullets.append([ny, x])
        bullets = new_bullets

        # Move blocks down.
        new_blocks = []
        for y, x in blocks:
            ny = y + 1
            if ny >= height:
                lives -= 1
                continue
            new_blocks.append([ny, x])
        blocks = new_blocks

        # Handle collisions (bullets vs blocks).
        hit = set()
        blocks_after = []
        for idx, (by, bx) in enumerate(blocks):
            if any(b for b in bullets if b[0] == by and b[1] == bx):
                hit.add(idx)
                score += 5
        blocks = [b for i, b in enumerate(blocks) if i not in hit]
        bullets = [b for b in bullets if (b[0], b[1]) not in {(blocks[i][0], blocks[i][1]) for i in range(len(blocks))}]

        # Player collision.
        for by, bx in blocks:
            if by == player_y and bx == player_x:
                lives -= 1
                blocks = [b for b in blocks if not (b[0] == by and b[1] == bx)]

        grid = [[EMPTY for _ in range(width)] for _ in range(height)]
        for y, x in blocks:
            grid[y][x] = BLOCK
        for y, x in bullets:
            if 0 <= y < height:
                grid[y][x] = BULLET
        grid[player_y][player_x] = PLAYER

        draw(stdscr, grid, score, lives)
        time.sleep(TICK)

    stdscr.nodelay(False)
    stdscr.addstr(height // 2, max(0, width // 2 - 5), "Game over. Press any key.")
    stdscr.getch()


def main() -> None:
    curses.wrapper(run_game)


if __name__ == "__main__":
    main()
