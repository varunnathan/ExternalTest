import gzip
import json
import re
from collections import defaultdict
from typing import List, Dict, Set
import sys
from nltk.stem import PorterStemmer


class AmazonMetaDataMatching:
    stemmer = PorterStemmer()

    def __init__(self):
        pass

    @staticmethod
    def get_gz_reader(file_path: str):
        return gzip.open(file_path, 'rt', encoding='utf-8')

    @staticmethod
    def get_gz_writer(file_path: str):
        return gzip.open(file_path, 'wt', encoding='utf-8')

    @staticmethod
    def get_indexed_string(text: str, word_to_index: Dict[str, int], stopwords: Set[str]) -> List[int]:
        text = re.sub(r'[^a-zA-Z ]', '', text).lower()
        tokens = text.split()  # Assuming a simple split for tokenization
        indexed_string = []
        for token in tokens:
            if stopwords and token in stopwords:
                continue
            stemmed_token = AmazonMetaDataMatching.stemmer.stem(token)
            if stemmed_token in word_to_index:
                indexed_string.append(word_to_index[stemmed_token])
        return indexed_string

    @staticmethod
    def main(args: List[str]):
        param_file = args[0]
        input_file = args[1]
        output_dir = args[2]

        fields = ["title", "description"]
        stopwords = None

        if param_file.lower() != "false":
            with open(param_file, 'r') as f:
                params = json.load(f)
                stopwords = set(params.get("stopwords", "").split())

        with AmazonMetaDataMatching.get_gz_reader(f"{output_dir}/product.txt.gz") as reader:
            products = [line.strip() for line in reader]

        product_to_index = {product: idx for idx, product in enumerate(products)}

        with AmazonMetaDataMatching.get_gz_reader(f"{output_dir}/vocab.txt.gz") as reader:
            vocab = [line.strip() for line in reader]

        word_to_index = {word: idx for idx, word in enumerate(vocab)}

        product_descriptions = {}
        product_categories = defaultdict(list)

        with AmazonMetaDataMatching.get_gz_reader(input_file) as reader:
            for line in reader:
                data = json.loads(line)
                asin = data["asin"]
                if asin not in product_to_index:
                    continue

                description = ""
                for field in fields:
                    if field in data:
                        if isinstance(data[field], list):
                            v = data[field][0] if data[field] else ""
                        else:
                            v = data[field]
                        description += v + " "

                product_descriptions[asin] = AmazonMetaDataMatching.get_indexed_string(description, word_to_index, stopwords)

                categories = data.get("categories", [])
                for category_list in categories:
                    category_string = " ".join(category_list)
                    indexed_category = AmazonMetaDataMatching.get_indexed_string(category_string, word_to_index, stopwords)
                    unique_indices = list(dict.fromkeys(indexed_category))
                    unique_indices.append(len(category_list))
                    product_categories[asin].append(unique_indices[::-1])

        with AmazonMetaDataMatching.get_gz_writer(f"{output_dir}/product_des.txt.gz") as writer:
            for product in products:
                indices = product_descriptions.get(product, [])
                writer.write(" ".join(map(str, indices)) + "\n")

        with AmazonMetaDataMatching.get_gz_writer(f"{output_dir}/product_query.txt.gz") as writer:
            for product in products:
                categories = product_categories.get(product, [])
                for category in categories:
                    writer.write(f"c{category[0]}\t" + " ".join(map(str, category[1:])) + ";\n")
                writer.write("\n")


if __name__ == "__main__":
    AmazonMetaDataMatching.main(sys.argv[1:])
