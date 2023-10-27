import pyautogui
import pygetwindow as gw
import cv2
import numpy as np
import random
import time
import pytesseract
import os
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Wczytanie szablonów
cell_template = cv2.imread('res/cell.png', 0)
mine_template = cv2.imread('res/mine.png', 0)
empty_template = cv2.imread('res/noMine.png', 0)
flag_template = cv2.imread('res/flag.png', 0)

print("cell_template:", cell_template.shape, cell_template.dtype)
print("mine_template:", mine_template.shape, mine_template.dtype)
print("empty_template:", empty_template.shape, empty_template.dtype)
print("flag_template:", flag_template.shape, flag_template.dtype)


if mine_template is None or empty_template is None or flag_template is None:
    print("Nie udało się wczytać jednego lub więcej szablonów.")
    exit()


def capture_screenshot(window_title, left_margin=10, top_margin=30, right_margin=10, bottom_margin=10):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        if window is not None:
            x, y, width, height = window._rect.left, window._rect.top, window.width, window.height
            
            # Dostosowanie współrzędnych i wymiarów
            screenshot = pyautogui.screenshot(region=(x + left_margin, y + top_margin, width - (left_margin + right_margin), height - (top_margin + bottom_margin)))
            
            screenshot_np = np.array(screenshot)
            screenshot_cv2 = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB)
            return screenshot_cv2
        else:
            print(f"Nie znaleziono okna o tytule {window_title}")
            return None
    except Exception as e:
        print(f"Wystąpił błąd podczas przechwytywania screenshotu: {e}")
        return None

def match_template(image, template):
    """
    Dopasowuje szablon do obrazu i zwraca współrzędne najlepszego dopasowania.
    
    Parametry:
    - image: obraz, na którym ma być przeprowadzone dopasowanie.
    - template: szablon, który ma być dopasowany.
    
    Zwraca:
    - tuple (x, y) reprezentujący współrzędne najlepszego dopasowania.
    """


    # Konwersja na skalę szarości, jeśli obraz jest kolorowy
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Konwersja na skalę szarości, jeśli szablon jest kolorowy
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Upewnienie się, że obraz jest większy niż szablon
    if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
        print("Obraz jest mniejszy niż szablon. Dopasowanie niemożliwe.")
        return None
    
    try:
        # Wykonywanie dopasowania
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        # Znalezienie pozycji najlepszego dopasowania
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Współrzędne najlepszego dopasowania
        top_left = max_loc
        x, y = top_left
        
        return (x, y)
    
    except cv2.error as e:
        print(f"Wystąpił błąd: {e}")
        return None


def analyze_board(screenshot):
    board = []
    cell_size = 50  
    
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    
    # Utworzenie katalogu 'cells', jeśli nie istnieje
    if not os.path.exists('cells'):
        os.makedirs('cells')
    
    for i in range(0, screenshot.shape[0], cell_size):
        row = []
        for j in range(0, screenshot.shape[1], cell_size):
            cell = screenshot[i:i + cell_size, j:j + cell_size]
            
            # Dopasowywanie cell_template
            cell_match_result = cv2.matchTemplate(cell, cell_template, cv2.TM_CCOEFF_NORMED)
            _, cell_match_val, _, _ = cv2.minMaxLoc(cell_match_result)

            
            # Zbieramy dopasowania dla różnych typów komórek
            match_results = {
                'M': cv2.matchTemplate(cell, mine_template, cv2.TM_CCOEFF_NORMED),
                'E': cv2.matchTemplate(cell, empty_template, cv2.TM_CCOEFF_NORMED),
                'F': cv2.matchTemplate(cell, flag_template, cv2.TM_CCOEFF_NORMED),
            }
            
            # Wybieramy najbardziej prawdopodobny typ komórki
            max_val = -1
            best_match = 'U'
            for cell_type, result in match_results.items():
                _, val, _, _ = cv2.minMaxLoc(result)
                if val > max_val and val > cell_match_val:  # Tylko jeśli wartość dopasowania jest wyższa niż dla cell_template
                    max_val = val
                    best_match = cell_type
                    
            # Jeśli żadne dopasowanie nie przekroczyło progu, użyj OCR
            if best_match == 'U':
                number = image_to_string(cell, config='--psm 6').strip()
                if number and number.isdigit():
                    best_match = number
                    
            row.append(best_match)
            
            # Zapisanie analizowanego skrawka z wartością threshold
            filename = f'cells/cell_{i//cell_size}_{j//cell_size}_{best_match}_{max_val:.2f}.png'
            cv2.imwrite(filename, cell)
            
        board.append(row)
        
    return board

def decide_next_move(board):
    # Wyszukaj puste komórki obok których znajdują się cyfry
    candidates = []
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'E':
                neighbors = get_neighbors(i, j, board)
                if any(cell.isdigit() for cell in neighbors):
                    candidates.append((i, j))

    # Jeżeli znaleziono takie komórki, jedna z nich będzie następnym ruchem
    if candidates:
        return random.choice(candidates)
    
    # Jeżeli nie znaleziono, wybierz losową pustą komórkę
    empty_cells = [(i, j) for i in range(len(board)) for j in range(len(board[0])) if board[i][j] == 'E']
    if empty_cells:
        return random.choice(empty_cells)

    # Jeżeli nie ma żadnych dostępnych ruchów, zwróć None
    return None

def get_neighbors(x, y, board):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            if 0 <= x + dx < len(board) and 0 <= y + dy < len(board[0]):
                neighbors.append(board[x + dx][y + dy])
    return neighbors

def make_move(x, y, cell_size):
    # Oblicz środek komórki
    screen_x = x * cell_size + cell_size // 2
    screen_y = y * cell_size + cell_size // 2
    
    # Ustal pozycję kursora
    pyautogui.moveTo(screen_x, screen_y)
    
    # Kliknij lewym przyciskiem myszy
    pyautogui.click()

def main():
    # Rozmiar komórki i planszy
    cell_size = 50
    board_size = (10, 10)  # W formie (szerokość, wysokość)

    # Odczekaj chwilę, żeby dać czas na przejście do okna gry
    print("Przejdź do okna gry w ciągu 5 sekund.")
    time.sleep(5)

    while True:
        # Zrób zrzut ekranu
        screenshot = capture_screenshot('Saper')

        # Analizuj planszę
        board = analyze_board(screenshot)
        result = decide_next_move(board)
        if result is None:
            print("Funkcja decide_next_move zwróciła None")
            # Możesz tu zdecydować, co zrobić w przypadku, gdy funkcja zwraca None.
            # Na przykład, możesz zakończyć program albo pominąć iterację pętli.
            return
        else:
            x, y = result
            make_move(x, y, cell_size)  # Ten kod zostanie wykonany tylko wtedy, gdy result nie jest None.

        # Czekaj na aktualizację gry po wykonanym ruchu
        time.sleep(1)  # Odczekaj 1 sekundę

        # (Opcjonalnie) Sprawdź, czy gra się zakończyła
        # if check_for_game_end():
        #     break

# Uruchom główną pętlę
if __name__ == "__main__":
    main()