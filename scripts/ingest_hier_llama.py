from app.services.llama_hier_ingest_service import ingest_pharma_data_hybrid


def main():
    result = ingest_pharma_data_hybrid("data/raw")
    print(result)


if __name__ == "__main__":
    main()