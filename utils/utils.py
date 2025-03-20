# utils/utils.py
register_flag = False

def set_register_flag(choice):
    global register_flag
    print("Setting register flag to", choice)
    register_flag = choice