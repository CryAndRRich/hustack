import os
import re
import numpy as np
import pandas as pd

class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        for col in ["id", "label"]:
            self.data[col] = self.data[col].fillna(-1)

        self.data["Question"] = self.data["Question"].astype(str)

    def check_stats(self):
        df = self.data[self.data["label"] != -1]
        counts = df["label"].value_counts().sort_index()
        total = counts.sum()
        percent = (counts / total * 100).round(2)
        stats = pd.DataFrame({"count": counts, "percent": percent})
        print("Label distribution (count and %):")
        print(stats)

    def __text_processed(self) -> None:
        def clean_math_text(text):
            text = str(text)
            
            math_symbols = {
                r"\$": " dollar_expr ",
                r"\=": " equals ",
                r"\<": " less than ",
                r"\>": " greater than ",
                r"\+": " plus ",
                r"\-": " minus ",
                r"\*": " times ",
                r"\/": " divided by ",
                r"\^": " to the power of ",
                r"\√": " square root ",
                r"\π": " pi_expr ",
                r"\∑": " sum_expr ",
                r"\∫": " integral_expr ",
                r"\∞": " infinity_expr "
            }
            
            for pattern, replacement in math_symbols.items():
                text = re.sub(pattern, replacement, text)
            
            text = re.sub(r"\\[a-zA-Z]+", " ", text)
            text = re.sub(r"\{([^}]*)\}", r" \1 ", text)
            
            text = re.sub(r"[^a-zA-Z0-9\s\.\?\!]", " ", text)
            
            text = re.sub(r"(\d+)([a-zA-Z])", r"\1 \2", text)  
            text = re.sub(r"([a-zA-Z])(\d+)", r"\1 \2", text)  
            
            text = re.sub(r"\s+", " ", text).strip().lower()
            
            return text
            
        self.data["Question"] = self.data["Question"].apply(clean_math_text)

    def __math_processed(self):
        self.data["length"] = self.data["Question"].astype(str).apply(len)
        self.data["log_length"] = np.log1p(self.data["length"])

        self.data["numeric_token_counts"] = self.data["Question"].str.findall(r"[\d+\-*/=()<>^√π÷%\.]").str.len()
        self.data["log_numeric_token_counts"] = np.log1p(self.data["numeric_token_counts"])
        
        def extract_math_features(text):
            features = {
                "num_count": len(re.findall(r"\d+", text)),
                "equation_count": len(re.findall(r"equals|equation|formula|solve for", text)),
                "function_count": len(re.findall(r"function|f\(x\)|derivative|integral", text)),
                "geometry_count": len(re.findall(r"angle|triangle|circle|area|volume", text)),
                "algebra_count": len(re.findall(r"variable|polynomial|matrix|vector", text)),
                "calculus_count": len(re.findall(r"derivative|integral|limit|differentiation", text)),
                "prob_count": len(re.findall(r"probability|random|random variable", text)),
                "topo_count": len(re.findall(r"ring|abelian|nonabelian", text)),
                "word_count": len(text.split()),
                "is_proof": int("prove" in text or "show that" in text),
                "is_compute": int("compute" in text or "calculate" in text),
                "is_find": int("find" in text or "determine" in text),
                "is_matrix": int("matrix" in text or "matrices" in text),
            }
            return pd.Series(features)

        math_features = self.data["Question"].apply(extract_math_features)
        self.data = pd.concat([self.data, math_features], axis=1)

    def data_processed(self) -> None:
        self.__text_processed()
        self.__math_processed()

    def save_csv(self) -> str:
        output_file = os.path.join(self.data_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"

if __name__ == "__main__":
    data_path = "math_prob/data"
    data = Data(data_path)

    data.data_processed()
    data.save_csv()
    data.check_stats()