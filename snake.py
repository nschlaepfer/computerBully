import random
import curses

def print_menu(stdscr, selected_row_idx):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    for idx, row in enumerate(['Play', 'High Scores', 'Settings', 'Quit']):
        x = w//2 - len(row)//2
        y = h//2 - 2 + idx
        if idx == selected_row_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(y, x, row)
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(y, x, row)
    stdscr.refresh()

def play_game(stdscr):
    try:
        curses.curs_set(0)
        sh, sw = stdscr.getmaxyx()
        w = curses.newwin(sh, sw, 0, 0)
        w.keypad(1)
        w.timeout(100)

        curses.start_color()
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)

        snk_x = sw//4
        snk_y = sh//2
        snake = [
            [snk_y, snk_x],
            [snk_y, snk_x-1],
            [snk_y, snk_x-2]
        ]

        food = [sh//2, sw//2]
        w.addch(int(food[0]), int(food[1]), curses.ACS_PI)

        obstacles = [[sh//2, sw//2 - 5], [sh//2, sw//2 - 4], [sh//2, sw//2 - 3]]
        for obstacle in obstacles:
            w.addch(int(obstacle[0]), int(obstacle[1]), '#')

        powerups = {
            'invincibility': {'symbol': 'I', 'duration': 10},
            'speed_boost': {'symbol': 'S', 'duration': 5}
        }

        active_powerups = {}

        key = curses.KEY_RIGHT

        score = 0

        while True:
            next_key = w.getch()
            key = key if next_key == -1 else next_key

            if snake[0][0] in [0, sh] or \
                snake[0][1]  in [0, sw] or \
                snake[0] in snake[1:] or \
                (snake[0] in obstacles and 'invincibility' not in active_powerups):
                break

            new_head = [snake[0][0], snake[0][1]]

            if key == curses.KEY_DOWN:
                new_head[0] += 1
            if key == curses.KEY_UP:
                new_head[0] -= 1
            if key == curses.KEY_LEFT:
                new_head[1] -= 1
            if key == curses.KEY_RIGHT:
                new_head[1] += 1

            snake.insert(0, new_head)

            if snake[0] == food:
                food = None
                while food is None:
                    nf = [
                        random.randint(1, sh-1),
                        random.randint(1, sw-1)
                    ]
                    food = nf if nf not in snake and nf not in obstacles else None
                w.addch(food[0], food[1], curses.ACS_PI)
                score += 1
            elif snake[0] in [list(powerup.keys()) for powerup in powerups.values()]:
                powerup_type = [k for k, v in powerups.items() if v['symbol'] == w.inch(snake[0][0], snake[0][1])][0]
                active_powerups[powerup_type] = powerups[powerup_type]['duration']
                powerups.pop(powerup_type)
            else:
                tail = snake.pop()
                w.addch(int(tail[0]), int(tail[1]), ' ')

            for powerup_type, duration in list(active_powerups.items()):
                active_powerups[powerup_type] -= 1
                if active_powerups[powerup_type] == 0:
                    active_powerups.pop(powerup_type)

            if 'invincibility' in active_powerups:
                w.attron(curses.color_pair(1))
                w.addch(int(snake[0][0]), int(snake[0][1]), curses.ACS_CKBOARD)
                w.attroff(curses.color_pair(1))
            elif 'speed_boost' in active_powerups:
                w.attron(curses.color_pair(2))
                w.addch(int(snake[0][0]), int(snake[0][1]), curses.ACS_CKBOARD)
                w.attroff(curses.color_pair(2))
                w.timeout(50)
            else:
                w.addch(int(snake[0][0]), int(snake[0][1]), curses.ACS_CKBOARD)
                w.timeout(100)

            w.addstr(0, 0, 'Score: {}'.format(score))

    except Exception as e:
        curses.endwin()
        print("An error occurred: {}".format(e))

def view_high_scores(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "High Scores:")
    stdscr.addstr(2, 0, "1. Player 1 - 1000")
    stdscr.addstr(3, 0, "2. Player 2 - 900")
    stdscr.addstr(4, 0, "3. Player 3 - 800")
    stdscr.refresh()
    stdscr.getch()

def adjust_settings(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Settings:")
    stdscr.addstr(2, 0, "1. Difficulty Level: Easy")
    stdscr.addstr(3, 0, "2. Game Speed: Normal")
    stdscr.addstr(4, 0, "3. Game Controls: Arrow Keys")
    stdscr.refresh()
    stdscr.getch()

def main(stdscr):
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    current_row_idx = 0
    print_menu(stdscr, current_row_idx)

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row_idx > 0:
            current_row_idx -= 1
        elif key == curses.KEY_DOWN and current_row_idx < 3:
            current_row_idx += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if current_row_idx == 0:
                play_game(stdscr)
            elif current_row_idx == 1:
                view_high_scores(stdscr)
            elif current_row_idx == 2:
                adjust_settings(stdscr)
            elif current_row_idx == 3:
                break

        print_menu(stdscr, current_row_idx)

    curses.endwin()

curses.wrapper(main)
