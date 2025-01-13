from san_0ai import SanAI
from distributed_server import start_distributed_server
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    if args.distributed:
        start_distributed_server(args.world_size)
    else:
        ai = SanAI()
        print("San_0AI est prêt à discuter! (tapez 'quit' pour sortir)")
        
        while True:
            user_input = input("Vous: ")
            if user_input.lower() == 'quit':
                break
                
            response = ai.process_input(user_input)
            print(f"San_0AI: {response}")
            
            # Apprentissage à partir de l'interaction
            conversation = f"User: {user_input}\nAssistant: {response}"
            loss = ai.learn_from_interaction(conversation)
            print(f"[Debug] Loss: {loss}")

if __name__ == "__main__":
    main()
