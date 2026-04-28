from app.db.Chroma_clientV2 import reset_collection

def main():
    reset_collection()
    print("Collection deleted successfully.")

# if __name__ == "__main__":
#     main()