from src.core.qdrant_client import client

def main():
    print("Collections:")
    print(client.get_collections())

    print("\nCollection info:")
    print(client.get_collection("documents"))

if __name__ == "__main__":
    main()
