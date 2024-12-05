from functions import set_seed


def params_for_SACG():
    # set_seed(random_seed_bool=False, seed=9922)
    # set_seed(random_seed_bool=False, seed=123)
    set_seed(random_seed_bool=True)
    # N = 40
    # N = 50
    # N = 100
    # N = 150
    # N = 200
    # N = 250
    N = 300
    # N = 400
    # N = 450
    # N = 500
    # N = 600
    # N = 620
    # N = 700
    # N = 750
    # N = 800
    # N = 850
    # N = 900
    # N = 1000
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-4.map'
    # img_dir = 'room-32-32-4.map'

    # img_dir = '10_10_my_rand.map'
    img_dir = 'random-32-32-10.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'random-64-64-20.map'
    # max_time = 20
    # max_time = 100
    # max_time = 200
    max_time = 10000
    # corridor_size = 20
    # corridor_size = 10
    # corridor_size = 5
    # corridor_size = 3
    # corridor_size = 2
    corridor_size = 1

    to_render: bool = True
    # to_render: bool = False

    # is_sacg: bool = False
    is_sacg: bool = True

    # to_check_paths: bool = True
    to_check_paths: bool = False

    # to_save = True
    to_save = False
    return N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save


def params_for_LMAPF():
    set_seed(random_seed_bool=False, seed=7389)
    # set_seed(random_seed_bool=False, seed=123)
    # set_seed(random_seed_bool=True)
    # N = 50
    # N = 100
    # N = 140
    # N = 150
    # N = 200
    N = 175
    # N = 170
    # N = 250
    # N = 300
    # N = 400
    # N = 500
    # N = 600
    # N = 620
    # N = 700
    # N = 750
    # N = 800
    # N = 850
    # N = 900

    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'maze-32-32-4.map'
    # img_dir = 'room-32-32-4.map'

    # img_dir = '15-15-two-rooms.map'
    # img_dir = '15-15-four-rooms.map'
    img_dir = '15-15-six-rooms.map'
    # img_dir = '15-15-eight-rooms.map'
    # img_dir = '10_10_my_rand.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'random-64-64-20.map'
    # max_time = 20
    max_time = 100
    # max_time = 200
    # max_time = 1000
    # corridor_size = 20
    # corridor_size = 10
    # corridor_size = 5
    # corridor_size = 3
    # corridor_size = 2
    corridor_size = 1

    is_sacg: bool = False
    # is_sacg: bool = True

    to_render: bool = True
    # to_render: bool = False

    # to_check_paths: bool = True
    to_check_paths: bool = False

    # to_save = True
    to_save = False

    return N, img_dir, max_time, corridor_size, to_render, to_check_paths, is_sacg, to_save
