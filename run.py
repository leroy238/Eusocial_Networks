from train import train

episodes, max_buffer, lr, gamma, minibatch, target_update, num_bees,hidden_dim, N, decay = 10000, 64, .001 ,.99, 32, 10000, 32, 128, 5, 0.95
train(10000, 64, .001 ,.99, 0.5, 32, 10000, 32, 128, 5, 0.95)


# import cProfile, pstats, io
# from train import train   # your function

# args = (2, 64, 0.001, 0.99, 0.5, 32, 10_000, 32, 128, 5, 0.95)   # <- adjust if needed

# pr = cProfile.Profile()
# pr.enable()
# train(*args)
# pr.disable()

# s = io.StringIO()
# pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(40)  # top-40 calls
# print(s.getvalue())