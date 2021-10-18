import time
from douzero.env.move_generator import MovesGener
from douzero.env.move_detector import get_move_type
from douzero.env import move_selector


EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}


def action_to_str(action):
    if len(action) == 0:
        return "Pass"
    else:
        return "".join([EnvCard2RealCard[card] for card in action])


def type_exist(mlist, type):
    if not isinstance(mlist, list):
        return False
    for item in mlist:
        if not isinstance(item, type):
            return False
    return True


def search_actions(my_cards, other_cards, path_list, rival_move=None, prev_moves=None):
    if prev_moves is None:
        my_cards.sort()
        other_cards.sort()
    my_gener = MovesGener(my_cards)
    other_gener = MovesGener(other_cards)
    other_bombs = other_gener.gen_type_4_bomb()
    other_bombs.extend(other_gener.gen_type_5_king_bomb())
    legal_move_tree = []
    rival_move_info = {}
    m_type_start = 1
    m_type_end = 15
    if rival_move is not None:
        if len(rival_move) > 0:
            rival_move_info = get_move_type(rival_move)
            m_type_start = rival_move_info["type"]
            m_type_end = m_type_start + 1
        else:
            rival_move = None

    for mtype in range(m_type_start, m_type_end):
        my_moves = my_gener.gen_moves_by_type(mtype)
        if len(my_moves) == 0:
            continue
        if mtype == 4:
            other_moves = other_bombs
        else:
            other_moves = other_gener.gen_moves_by_type(mtype)
        for move in my_moves:
            if len(move) != len(my_cards):
                if mtype != 4 and mtype != 5 and len(other_bombs) > 0:
                    break
                if len(move_selector.filter_type_n(mtype, other_moves, move)) == 0:
                    if rival_move is not None:
                        move_info = get_move_type(move)
                        if move_info["rank"] <= rival_move_info["rank"]:
                            continue
                        if "len" in move_info and move_info["len"] != rival_move_info["len"]:
                            continue
                    new_cards = my_cards.copy()
                    for card in move:
                        new_cards.remove(card)
                    if prev_moves is not None:
                        new_prev = prev_moves.copy()
                        new_prev.append(move)
                    else:
                        new_prev = [move]
                    actions = search_actions(new_cards, other_cards, path_list, prev_moves=new_prev)
                    if actions is not None and len(actions) > 0:
                        legal_move_tree.append([move, [actions]])
            else:
                if rival_move is not None:
                    move_info = get_move_type(move)
                    if move_info["rank"] <= rival_move_info["rank"]:
                        continue
                    if "len" in move_info and move_info["len"] != rival_move_info["len"]:
                        continue
                legal_move_tree.append(move)
                if prev_moves is not None:
                    new_path = prev_moves.copy()
                    new_path.append(move)
                    path_list.append(new_path)
                else:
                    path_list.append([move])
    legal_moves_count = len(legal_move_tree)
    if legal_moves_count == 0:
        return None
    if legal_moves_count == 1:
        return legal_move_tree[0]
    return legal_move_tree


def eval_path(path):
    bomb = 0
    for action in path:
        if 30 in action and 20 in action or len(action) == 4 and len(set(action)) == 1:
            bomb += 1
    return 1 + bomb - len(path) * 0.05


def select_optimal_path(path_list):
    if len(path_list) != 0:
        max_path = max(path_list, key=lambda x: eval_path(x))
        for action in max_path:
            action.sort()
        return max_path
    else:
        return None


def check_42(path):
    for action in path:
        move_type = get_move_type(action)
        if move_type["type"] == 13 or move_type["type"] == 14:
            return True
    return False


if __name__ == "__main__":
    my_cards = [3, 3, 4, 5, 5, 6, 6, 6, 7, 8, 8, 10, 10, 14, 17, 17]
    other_cards = [11, 4]
    st = time.time()
    paths = []
    result = search_actions(my_cards, other_cards, paths, rival_move=[11, 11])
    print(time.time()-st)
    print(result)
    print(paths)
    path = select_optimal_path(paths)
    print(path)