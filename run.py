import argparse
from train import train

def main():
    parser = argparse.ArgumentParser(description="Train model with specified parameters.")
    parser.add_argument("--episodes", type=int, required=True, help="Number of episodes")
    parser.add_argument("--max_buffer", type=int, required=True, help="Max buffer size")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--gamma", type=float, required=True, help="Discount factor gamma")
    parser.add_argument("--minibatch", type=int, required=True, help="Minibatch size")
    parser.add_argument("--target_update", type=int, required=True, help="Target update frequency")
    parser.add_argument("--num_bees", type=int, required=True, help="Number of bees")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden layer dimension")
    parser.add_argument("--N", type=int, required=True, help="N value")
    parser.add_argument("--decay", type=float, required=True, help="Decay rate")
    parser.add_argument("--no_com", choices=[0,1],required=True, help="Use comm if 1")

    args = parser.parse_args()

    train(
        args.episodes,
        args.max_buffer,
        args.lr,
        args.gamma,
        args.minibatch,
        args.target_update,
        args.num_bees,
        args.hidden_dim,
        args.N,
        args.decay,
        True if args.no_com == 1 else False
    )

if __name__ == "__main__":
    main()
