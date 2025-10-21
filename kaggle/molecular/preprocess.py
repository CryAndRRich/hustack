import os
import random
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    def __drop_uninformative(self) -> None:
        cols_to_keep = self.mi_scores_dataset().index.tolist()
        cols_to_keep += ["Batch_ID", "T80", "Smiles"]
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        X = self.data[self.data["Batch_ID"].str.contains("Train")].copy()
        y = X.pop("T80")
        X.pop("Batch_ID")
        X.pop("Smiles")
            
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        mi_scores = mi_scores[mi_scores > 0]

        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        X = self.data[self.data["Batch_ID"].str.contains("Train")].copy()
        y = X.pop("T80")
        X.pop("Batch_ID")
        X.pop("Smiles")
        
        log_y = np.log(y)
        
        rmsle = np.sqrt(-cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_squared_error"
        ).mean())
        
        mae = -cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_absolute_error"
        ).mean()
        
        print(f"Baseline RMSLE: {rmsle:.5f}")
        print(f"Baseline MAE (log scale): {mae:.5f}")

    def __energy_processed(self) -> None:
        T_cols = [f"T{i}" for i in range(1, 21)]
        S_cols = [f"S{i}" for i in range(1, 21)]
        O_cols = [f"O{i}" for i in range(1, 21)]
        
        self.data["T_mean"] = self.data[T_cols].mean(axis=1)
        self.data["T_std"]  = self.data[T_cols].std(axis=1)
        self.data["T_min"]  = self.data[T_cols].min(axis=1)
        self.data["T_max"]  = self.data[T_cols].max(axis=1)
        self.data["T_range"] = self.data["T_max"] - self.data["T_min"]
        
        self.data["S_mean"] = self.data[S_cols].mean(axis=1)
        self.data["S_std"]  = self.data[S_cols].std(axis=1)
        self.data["S_min"]  = self.data[S_cols].min(axis=1)
        self.data["S_max"]  = self.data[S_cols].max(axis=1)
        self.data["S_range"] = self.data["S_max"] - self.data["S_min"]
        
        self.data["min_diff_S_T"] = self.data["S_min"] - self.data["T_min"]
        
        self.data["O_sum"] = self.data[O_cols].sum(axis=1)
        self.data["O_mean"] = self.data[O_cols].mean(axis=1)
        self.data["O_std"] = self.data[O_cols].std(axis=1)
    
    def __smiles_processed(self) -> None:
        def gen_rdkit(df):
            smiles = df["Smiles"]
            RDKitProp = np.zeros(((smiles.shape[0]), 10), dtype=float)
            for d_key, smile in enumerate(smiles):
                mol3d = Chem.AddHs(Chem.MolFromSmiles(smile))

                AllChem.EmbedMolecule(mol3d, randomSeed=random.randint(1, 1000000))
                AllChem.MMFFOptimizeMolecule(mol3d)

                RDKitProp[d_key, 8] = rdMolDescriptors.CalcNumRotatableBonds(mol3d)   # Number of rotatable bonds
            
            df = pd.concat([df, pd.DataFrame(RDKitProp, columns=[f"RDKit_{i}" for i in range(10)])], axis=1)
            return df

        self.data = gen_rdkit(self.data)
        
        def compute_molecular_complexity(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.nan, np.nan, np.nan
            
            ring_count = Descriptors.RingCount(mol)
            num_atoms = mol.GetNumAtoms()
            logP = Descriptors.MolLogP(mol)
            
            complexity = ring_count * 0.5 + num_atoms * 0.3 + logP * 0.2
            
            return ring_count, num_atoms, complexity

        self.data[["ring_count", "num_atoms", "complexity"]] = self.data["Smiles"].apply(
            lambda x: pd.Series(compute_molecular_complexity(x))
        )

        self.data["size_index"] = self.data["Mass"] / (self.data["Rg"] + 1e-6)  # Avoid division by zero
        self.data["hetero_polarity"] = self.data["NumHeteroatoms"] * self.data["TPSA"]
        self.data["Ncomplexity"] = self.data["NumHeteroatoms"] + self.data["Rg"] * 0.5 + self.data["TPSA"] * 0.1

    def __physical_processed(self) -> None:
        self.data["ratio_HDonors_HAcceptors"] = self.data["HDonors"] / (self.data["HAcceptors"] + 1e-6)
        self.data["mass_minus_logP"] = self.data["Mass"] - self.data["LogP"]
        self.data["total_H_bonds"] = self.data["HDonors"] + self.data["HAcceptors"]
    
    def __orbital_processed(self) -> None:
        self.data["energy_gap"] = self.data["LUMO(eV)"] - self.data["HOMO(eV)"]

        self.data["delta_LUMO"] = self.data["LUMOp1(eV)"] - self.data["LUMO(eV)"]
        self.data["delta_HOMO"] = self.data["HOMO(eV)"] - self.data["HOMOm1(eV)"]

        self.data["ratio_LUMO"] = self.data["delta_LUMO"] / (self.data["energy_gap"] + 1e-6)
        self.data["ratio_HOMO"] = self.data["delta_HOMO"] / (self.data["energy_gap"] + 1e-6)

        self.data["hardness"] = self.data["energy_gap"] / 2.0
        self.data["softness"] = 1 / (self.data["hardness"] + 1e-6)

        self.data["chemical_potential"] = (self.data["HOMO(eV)"] + self.data["LUMO(eV)"]) / 2.0

        self.data["electrophilicity"] = (
            (self.data["chemical_potential"] * self.data["chemical_potential"])
            / (2.0 * self.data["hardness"])
        )

        self.data["combined_orbital_feature"] = self.data["delta_LUMO"] - self.data["delta_HOMO"]

    def __tricky_predict(self) -> None:
        self.data["predict"] = (
            (self.data["T8"] - self.data["T2"] * self.data["TDOS3.2"] * self.data["LUMOp1(eV)"]) * (1.7020332667085079 * self.data["SDOS3.9"])
            +
            (
                ((self.data["TDOS3.7"] - self.data["SDOS3.8"] - self.data["TDOS4.3"] - self.data["TDOS2.7"])
                / ((self.data["T14"] / self.data["TDOS3.6"]) + (0.8998248512734 - self.data["RDKit_8"])))
                +
                ((self.data["LUMO(eV)"] + self.data["T2"] * self.data["TDOS3.2"] * self.data["LUMOp1(eV)"]) * ((self.data["LUMO(eV)"] + self.data["T2"] * self.data["TDOS3.2"] * self.data["LUMOp1(eV)"]) - self.data["TDOS3.2"]))
            )
        )
        
    def data_processed(self) -> None:
        self.__energy_processed()
        self.__smiles_processed()
        self.__physical_processed()
        self.__orbital_processed()
        self.__tricky_predict()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        self.output = "processed_data.csv"
        output_file = os.path.join(self.data_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    data_path = "molecular/data"
    data = Data(data_path)
    
    data.data_processed()
    data.baseline_score_dataset()
    data.save_csv()