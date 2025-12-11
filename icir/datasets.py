import re
import shutil
from enum import Enum
from functools import partial
from pathlib import Path

import anndata as ad
import datalair
import numpy as np
import pandas as pd

import icir._utils as utils


class ImmuneCheckpointTherapyResponse(datalair.Dataset):

    uuid = datalair.UUID("88d9a02deae6282b")

    def derive(self, lair: datalair.Lair) -> None:
        output_dir = lair.get_path(self)
        source_dir = Path("/fast-storage/tam/manual-download/ImmuneCheckpointTherapyResponse")
        assert source_dir.exists(), "Path {} does not exist".format(source_dir)
        for item in source_dir.iterdir():
            target_path = output_dir.joinpath(item.name)
            if item.is_dir():
                shutil.copytree(item, target_path)
            else:
                shutil.copy2(item, target_path)
        datalair.download_supplementary_from_geo("GSE115821", output_dir.joinpath("Auslander"))
        datalair.download_files_from_arrayexpress("E-MTAB-3218", output_dir.joinpath("Choueiri"))
        datalair.download_supplementary_from_geo("GSE78220", output_dir.joinpath("Hugo"))
        datalair.download_supplementary_from_geo("GSE100797", output_dir.joinpath("Lauss"))
        datalair.download_supplementary_from_geo("GSE91061", output_dir.joinpath("Riaz"))
        datalair.download_supplementary_from_geo("GSE176307", output_dir.joinpath("Rose"))
        datalair.download_supplementary_from_geo("GSE35640", output_dir.joinpath("Ulloa-Montoya"))


class ImmuneCheckpointTherapyResponseProcessed(datalair.Dataset):

    uuid = datalair.UUID("6a7266d1dc4017ac")

    def derive(self, lair: datalair.Lair) -> None:

        def process_auslander(filepaths):
            data = pd.read_csv(filepaths["Auslander"].joinpath("GSE115821_MGH_counts.csv.gz")).set_index("Geneid").T
            data = data.drop(columns=['1-Mar', '2-Mar'])  # excel error: genes were recognized as dates (MARCH1, MARCH2)
            var = data.iloc[:4].T
            data = data.iloc[5:]
            data.index = [sample[:-4] for sample in data.index]
            # obs_previously_published = pd.read_excel(filepaths["Auslander"].joinpath("41591_2018_157_MOESM6_ESM.xlsx"), skiprows=2, nrows=257) # this is a collection of clinical data from other studies
            obs_mgh_dataset = pd.read_excel(filepaths["Auslander"].joinpath("41591_2018_157_MOESM6_ESM.xlsx"),
                                            skiprows=263, nrows=37)  # this is their own dataset
            obs_mgh_dataset["Sample"] = [sample.split(".")[0].strip() for sample in obs_mgh_dataset["Sample"]]
            obs_mgh_dataset = obs_mgh_dataset.set_index("Sample")
            samples_with_ici_response = sorted(list(set(data.index).intersection(set(obs_mgh_dataset.index))))
            data = data.loc[samples_with_ici_response]
            obs_mgh_dataset = obs_mgh_dataset.loc[samples_with_ici_response]
            obs_mgh_dataset["original.sample_id"] = obs_mgh_dataset.index
            adata = ad.AnnData(data)
            assert np.all(adata.var.index == var.index)
            adata.var = var
            assert np.all(adata.obs.index == obs_mgh_dataset.index)
            adata.obs = obs_mgh_dataset
            return adata

        def process_chen(filpaths):
            data = pd.read_excel(filepaths["Chen"].joinpath("21598290cd151545-sup-160314_2_supp_0_f8k9tp.xlsx"),
                                 sheet_name="Supplementary Table S6a", skiprows=1, header=0, nrows=54)
            data["Patient ID (int)"] = data["Patient ID"]
            data.loc[data["Patient ID (int)"] == "•15", "Patient ID (int)"] = "15"
            data["Patient ID (int)"] = data["Patient ID (int)"].astype(int)
            data = data.sort_values("Patient ID (int)").reset_index()
            data = data[['Patient ID (int)'] + [col for col in data.columns if col != 'Patient ID (int)']]
            obs = data.iloc[:, :7]
            data = data.iloc[:, 7:]
            adata = ad.AnnData(data)
            adata.obs = obs
            return adata

        def process_freeman(filpaths):
            data = pd.read_excel(filepaths["Freeman"].joinpath("Supplementary_Table_4.xlsx"),
                                 sheet_name="7. Post_batch_correction", skiprows=0, header=0, nrows=None, index_col=0).T
            data = data.iloc[116:]
            data.index = [id.split("-")[0] for id in data.index]
            clinical_data = pd.read_excel(filepaths["Freeman"].joinpath("Supplementary_Table_1.xlsx"),
                                          sheet_name="6. MGH (RNA)", skiprows=1, header=0, nrows=88).ffill()
            data_with_duplicates = []
            # to_drop = []
            for patient in clinical_data["Patient"]:
                try:
                    data_with_duplicates.append(data.loc[patient])
                except KeyError:
                    clinical_data = clinical_data.loc[clinical_data["Patient"] != patient]
            data_with_duplicates = pd.concat(data_with_duplicates, axis=1).T
            data = ad.AnnData(data_with_duplicates)
            data.obs.reset_index(inplace=True)
            clinical_data.reset_index(inplace=True)
            data.obs = clinical_data
            return data

        def process_gide(filpaths):
            data = pd.read_excel(filepaths["Freeman"].joinpath("Supplementary_Table_6.xlsx"),
                                 sheet_name="2. Pre_batch_correction", skiprows=0, header=0, nrows=None,
                                 index_col=0).T.iloc[107:, :]
            obs_monotherapy_pd1 = pd.read_excel(filepaths["Gide"].joinpath("1-s2.0-S1535610819300376-mmc2.xlsx"),
                                                sheet_name="Table S1. PD-1 Patient", skiprows=2, index_col=0)
            obs_monotherapy_pd1["original.patient_id"] = obs_monotherapy_pd1.index
            obs_monotherapy_pd1["original.sample_id"] = np.nan
            obs_combined_pd1_ctla4 = pd.read_excel(filepaths["Gide"].joinpath("1-s2.0-S1535610819300376-mmc3.xlsx"),
                                                   sheet_name="Table S2. PD-1+CTLA-4 patients", skiprows=2, index_col=0)
            obs_combined_pd1_ctla4["original.patient_id"] = obs_combined_pd1_ctla4.index
            obs_combined_pd1_ctla4["original.sample_id"] = np.nan
            clinical_data = []
            for sample_id in data.index:
                patient_number, ici_target, _ = sample_id.split("_")
                patient_number = int(patient_number)
                match ici_target:
                    case "PD1":
                        obs_monotherapy_pd1.loc[patient_number, "original.sample_id"] = sample_id
                        clinical_data.append(obs_monotherapy_pd1.loc[patient_number])
                    case "ipiPD1":
                        obs_combined_pd1_ctla4.loc[patient_number, "original.sample_id"] = sample_id
                        clinical_data.append(obs_combined_pd1_ctla4.loc[patient_number])
                    case _:
                        raise Exception(f"Unknown ICI target: {ici_target}")
            clinical_data = pd.DataFrame(clinical_data)
            clinical_data.index = data.index
            data = ad.AnnData(data)
            data.obs = clinical_data
            return data

        def process_hugo(filpaths):
            # TODO: Check again RECIST vs. Tretment response. There are insconsitencies???
            data = pd.read_excel(filepaths["Hugo"].joinpath("GSE78220_PatientFPKM.xlsx"), index_col=0).T
            data.index = [patient.replace(".", "-") for patient in data.index]
            data = data.drop("Pt16-OnTx")  # no clinical data on this patient
            clinical_data = pd.read_csv(filepaths["Hugo"].joinpath("Hugo_clinical.tsv"), sep="\t", index_col=0)
            clinical_data_from_freeman = pd.read_excel(filepaths["Freeman"].joinpath("Supplementary_Table_1.xlsx"),
                                                       sheet_name="9. Hugo et al. (RNA)", skiprows=1)
            clinical_data_from_freeman.index = [patient + "-baseline" for patient in
                                                clinical_data_from_freeman["Patient"]]
            clinical_data_from_freeman = clinical_data_from_freeman.add_prefix("Freeman.")
            clinical_data = pd.concat([clinical_data, clinical_data_from_freeman], axis=1)
            clinical_data.loc["Pt27A-baseline"] = clinical_data.loc["Pt27-baseline"]
            clinical_data.loc["Pt27B-baseline"] = clinical_data.loc["Pt27-baseline"]
            clinical_data = clinical_data.drop("Pt27-baseline")
            assert np.all(data.sort_index().index == clinical_data.sort_index().index)
            data = ad.AnnData(data)
            data.obs = clinical_data
            return data

        def process_lauss(filepaths):
            # TODO: TIL Therapies??? Not ICI therapies???
            data = pd.read_csv(filepaths["Lauss"].joinpath("GSE100797_ProcessedData.txt.gz"), sep="\t", index_col=0).T
            data.index = [index[:-2] for index in data.index]
            clinical_data = pd.read_excel(filepaths["Lauss"].joinpath("41467_2017_1460_MOESM4_ESM.xlsx"), skiprows=1,
                                          index_col=0)
            clinical_data = clinical_data[clinical_data["RNAseq"] == "yes"]
            clinical_data["original.patient_id"] = clinical_data.index
            clinical_data["original.sample_id"] = clinical_data.index
            assert np.all(clinical_data.index == data.index)
            data = ad.AnnData(data)
            data.obs = clinical_data
            return data

        def process_liu(filepaths):
            data = pd.read_csv(filepaths["Liu"].joinpath("41591_2019_654_MOESM3_ESM.txt"), sep="\t", index_col=0)
            clinical_data_from_freeman = pd.read_excel(filepaths["Freeman"].joinpath("Supplementary_Table_1.xlsx"),
                                                       sheet_name="10. Liu et al. (RNA)", skiprows=1)
            clinical_data_from_freeman["Patient ID"] = [patient.split("_")[0] for patient in
                                                        clinical_data_from_freeman["Patient"]]
            clinical_data_from_freeman = clinical_data_from_freeman.drop(columns=["Age", "Gender", "Primary_tumor"])
            assert clinical_data_from_freeman["Patient ID"].nunique() == len(clinical_data_from_freeman)
            clinical_data_from_freeman = clinical_data_from_freeman.set_index("Patient ID")
            clinical_data_from_freeman["original_sample_id"] = clinical_data_from_freeman.index
            data = data.loc[clinical_data_from_freeman.index]
            adata = ad.AnnData(data)
            adata.obs = clinical_data_from_freeman
            return adata

        def process_prat(filepaths):
            data = pd.read_excel(
                filepaths["Prat"].joinpath("00085472can163556-sup-176636_2_unknown_upload_3997093_ddtlnx.xls"),
                sheet_name="raw data", index_col=0).T
            obs = pd.read_excel(
                filepaths["Prat"].joinpath("00085472can163556-sup-176636_2_unknown_upload_3997093_ddtlnx.xls"),
                sheet_name="clinical data", index_col=0)
            obs["original.sample_id"] = obs.index
            obs["original.patient_id"] = obs.index
            var = data.loc[["Accession", "Class Name"]].T
            data = data.drop(["Accession", "Class Name"])
            assert np.all(data.index == obs.index)
            data = ad.AnnData(data)
            data.var = var
            data.obs = obs
            return data

        def process_ravi(filepaths):
            data = pd.read_excel(filepaths["Ravi"].joinpath("41588_2023_1355_MOESM3_ESM.xlsx"),
                                 sheet_name="Table_S13_RNA_TPM", index_col=0, skiprows=2)
            var = data["Description"]
            data = data.T.drop("Description")
            obs = pd.read_excel(filepaths["Ravi"].joinpath("41588_2023_1355_MOESM3_ESM.xlsx"),
                                sheet_name="Table_S1_Clinical_Annotations", skiprows=2)
            obs = obs.dropna(subset=["Harmonized_SU2C_RNA_Tumor_Sample_ID_v2"])
            assert np.all(obs["Harmonized_SU2C_RNA_Tumor_Sample_ID_v2"] == data.index)
            data = ad.AnnData(data)
            data.var["Description"] = list(var)
            data.obs = obs
            return data

        def process_riaz(filepaths):
            # Riaz (Entrez Gene IDs)
            data = pd.read_csv(filepaths["Riaz"].joinpath("GSE91061_BMS038109Sample.hg19KnownGene.raw.csv.gz"),
                               index_col=0).T
            clinical_data = pd.read_excel(filepaths["Riaz"].joinpath("mmc2.xlsx"), sheet_name="Table S2", skiprows=2,
                                          index_col=0)
            clinical_data["origional.patient_id"] = clinical_data.index
            clinical_data_per_rna_seq_sample = {}
            for sample_id in data.index:
                patient_number = sample_id.split("_")[0]
                if patient_number in clinical_data.index:
                    clinical_data_per_rna_seq_sample[sample_id] = clinical_data.loc[patient_number]
            clinical_data_per_rna_seq_sample = pd.DataFrame(clinical_data_per_rna_seq_sample).T
            clinical_data_per_rna_seq_sample["original.sample_id"] = clinical_data_per_rna_seq_sample.index
            obs = clinical_data_per_rna_seq_sample
            data = ad.AnnData(data.loc[obs.index])
            data = ad.AnnData(data)
            data.obs = obs
            return data

        def process_rose(filepaths):
            data = pd.read_csv(filepaths["Rose"].joinpath("GSE176307_salmon_tpm_gene.matrix.tsv.gz"), sep="\t",
                               index_col=0).T
            clinical_data = pd.read_csv(filepaths["Rose"].joinpath("Rose_clinical.tsv"), sep="\t", index_col=0)
            data = ad.AnnData(data.loc[clinical_data["Sample_id"]])
            data.obs = clinical_data
            return data

        def process_snyder(filepaths):
            clinical_data = pd.read_csv(filepaths["Snyder"].joinpath("data_clinical.csv"))
            clinical_data["my_patient_id"] = [int(sample_id.split("_")[0]) for sample_id in
                                           clinical_data["sample_id_dna_normal"]]
            clinical_data = clinical_data.set_index("my_patient_id")
            clinical_data_from_shen = pd.read_csv(filepaths["Snyder"].joinpath("Snyder_clinical.tsv"), sep="\t")
            clinical_data_from_shen = clinical_data_from_shen.set_index("patient_id")
            clinical_data_from_shen = clinical_data_from_shen.add_prefix("shen.")
            clinical_data = pd.concat([clinical_data, clinical_data_from_shen], axis=1)

            tpms = []
            for patient_id, bam_id in clinical_data["bam_id_rna_tumor"].items():
                filepath = filepaths["Snyder"].joinpath("kallisto", bam_id + "-kallisto", "abundance.tsv")
                assert filepath.exists()
                data = pd.read_csv(filepath, sep="\t", index_col=0)["tpm"]
                data.name = patient_id
                tpms.append(data)
            data = ad.AnnData(pd.concat(tpms, axis=1).T)
            data.obs = clinical_data
            return data

        def process_vanallen(filepaths):
            data = pd.read_excel(filepaths["Freeman"].joinpath("Supplementary_Table_4.xlsx"),
                                 sheet_name="7. Post_batch_correction", index_col=0).T
            data = data.loc[[id for id in data.index if id[:11] == "MEL.IPI_Pat"]]
            clinical_data = pd.read_excel(filepaths["VanAllen"].joinpath("tables2_revised.xlsx"),
                                          sheet_name="exome analysis (n=110)", index_col=0)
            clinical_data = clinical_data.loc[[id.split(".")[1][4:] for id in data.index]]
            clinical_data["original.patient_id"] = clinical_data.index
            clinical_data["original.sample_id"] = data.index
            data = ad.AnnData(data)
            data.obs = clinical_data
            return data

        datasets = {
            "Auslander": process_auslander,
            "Chen": process_chen,
            # Choueiri TODO: get data
            "Freeman": process_freeman,
            "Gide": process_gide,
            "Hugo": process_hugo,
            "Lauss": process_lauss,
            "Liu": process_liu,
            "Prat": process_prat,
            "Ravi": process_ravi,
            "Riaz": process_riaz,
            "Rose": process_rose,
            "Snyder": process_snyder,
            # Ulloa-Montoya (This dataset is on MAGE-A3) TODO: Get data
            "VanAllen": process_vanallen
            # Zhao TODO: Download and process fastq files
        }

        output_dir = lair.get_path(self)
        ds = ImmuneCheckpointTherapyResponse()
        lair.safe_derive(ds)
        filepaths = lair.get_dataset_filepaths(ds)
        for name, func in datasets.items():
            adata = func(filepaths)
            adata.X = np.array(adata.X, dtype=np.float32)
            for col in adata.obs.columns:
                if not pd.api.types.is_numeric_dtype(adata.obs[col]):
                    adata.obs[col] = adata.obs[col].astype(str)
            for col in adata.var.columns:
                if not pd.api.types.is_numeric_dtype(adata.var[col]):
                    adata.var[col] = adata.var[col].astype(str)
            adata.write(output_dir.joinpath("{}.h5ad".format(name)))


class ImmuneCheckpointTherapyResponseProcessedGeneNormalized(datalair.Dataset):

    def derive(self, lair: datalair.Lair) -> None:

        def sum_reduce_duplicated_genes(adata):
            duplicated_genes_mask = adata.var.index.duplicated("first")
            new_adata = adata[:, ~duplicated_genes_mask].copy()
            redundant_adata = adata[:, duplicated_genes_mask].copy()
            for gene in redundant_adata.var_names:
                new_adata[:, gene].X = new_adata[:, gene].X + redundant_adata[:, gene].X.sum(axis=1)[:, np.newaxis]
            return new_adata

        def normalize_adata_genes(adata, pre_id_transform):
            genes = utils.normalize_all_genes(adata.var_names, pre_id_transform)
            genes = genes.dropna().sort_values("gene_id")
            adata = adata[:, genes.dropna().index]
            adata.var = genes
            adata.var.set_index("gene_id", drop=True, inplace=True)
            adata = sum_reduce_duplicated_genes(adata)
            return adata

        def process_snyder(adata):
            genes = utils.normalize_all_genes(adata.var_names, "transcript")
            genes = genes.dropna().sort_values("gene_id")
            adata = adata[:, genes.dropna().index]
            adata.var = genes
            adata.var.reset_index(inplace=True)
            bdata = ad.AnnData(pd.DataFrame(
                adata.var.groupby("gene_id").apply(lambda x: pd.Series(adata.X[:, x.index].sum(axis=1)))).T)
            bdata.var = adata.var.groupby(
                [col for col in adata.var.drop(columns="index").columns if col != 'original.id'], as_index=False).agg(
                {'original.id': list}).set_index("gene_id", drop=True).sort_index()
            bdata.obs = adata.obs
            return bdata

        def process_rose(adata):
            adata = normalize_adata_genes(adata, "hugo")
            adata.var = adata.var[adata.var.columns[1:].tolist() + [adata.var.columns[0]]]
            return adata

        output_dir = lair.get_path(self)

        ds = ImmuneCheckpointTherapyResponseProcessed()
        lair.safe_derive(ds, overwrite=False)
        filepaths = lair.get_dataset_filepaths(ds)

        for name, path in filepaths.items():
            adata = ad.read(path)
            match name.lower().split(".")[0]:
                case "freeman" | "ravi" | "gide" | "vanallen":
                    adata = normalize_adata_genes(adata, None)
                case "riaz":
                    adata = normalize_adata_genes(adata, "entrez")
                case "chen" | "lauss" | "auslander" | "liu" | "hugo" | "prat":
                    adata = normalize_adata_genes(adata, "hugo")
                case "snyder":
                    print(name, adata)
                    adata = process_snyder(adata)
                case "rose":
                    adata = process_rose(adata)
                case _:
                    raise ValueError("Uknown dataset: {}".format(name))
            for col in adata.obs.select_dtypes(include="object").columns:
                adata.obs[col] = adata.obs[col].astype(str)
            for col in adata.var.select_dtypes(include="object").columns:
                adata.var[col] = adata.var[col].astype(str)
            adata.write(output_dir.joinpath(name))


class ImmuneCheckpointTherapyResponseProcessedGeneNormalizedClinicalDataNormalized(datalair.Dataset):

    def derive(self, lair) -> None:

        def normalize_values_of_clinical_data(df: pd.DataFrame) -> pd.DataFrame:
            df["age"] = df["age"].astype("Int64")
            df["sex"] = df["sex"].map(lambda x: {"F": "female", "M": "male", "nan": np.nan}.get(x, x))
            df["overall_survival"] = np.round(df["overall_survival"], decimals=0).astype("Int64")
            df["response"] = df["response"].map(lambda x: {"nan": np.nan}.get(x, x))
            df["RECIST"] = df["RECIST"].map(lambda x: {"PD ": "PD", "NE": np.nan, "nan": np.nan, "SD ": "SD", "X": np.nan}.get(x, x))
            df["prior_CTLA4"] = df["prior_CTLA4"].astype("boolean")
            df["prior_PD1"] = df["prior_PD1"].astype("boolean")
            df["timing"] = df["timing"].map(lambda x: "Post" if re.fullmatch(r"Post-I{0,3}", x) else x)
            return df

        recist_to_response_map = {"SD": "NR", "PD": "NR", "PR": "R", "CR": "R"}

        def ici_agent_to_ici_target(ici_agent):
            ici_targets = set()
            if re.match(r"Nivolumab|Pembrolizumab", ici_agent, re.IGNORECASE):
                ici_targets.add("PD-1")
            if re.match(r"Durvalumab|Avelumab", ici_agent, re.IGNORECASE):
                ici_targets.add("PD-L1")
            if re.match(r"Ipilimumab|Tremelimumab|Atezolizumab", ici_agent, re.IGNORECASE):
                ici_targets.add("CTLA-4")
            if re.match(r"Lirilumab", ici_agent, re.IGNORECASE):
                ici_targets.add("KIR")
            if re.match(r"Urelumab", ici_agent, re.IGNORECASE):
                ici_targets.add("CD137")
            if re.match(r"Epacadostat", ici_agent, re.IGNORECASE):
                ici_targets.add("IDO")
            if re.match(r"LAG-3", ici_agent, re.IGNORECASE):
                ici_targets.add("LAG-3")
            if re.match(r"Carboplatin|Pemetrexed|Gemcitabine", ici_agent, re.IGNORECASE):
                ici_targets.add("Cytostasis")
            return ici_targets

        def flag_seen(group, target_treatment):
            seen = False
            flags = []
            for treatment in group['Therapy']:
                flags.append(seen)
                if treatment == target_treatment:
                    seen = True
            return flags

        class Column(Enum):
            AGE = "age"  # in years
            SEX = "sex"  # male, female
            PRIMARY_TUMOR = "primary_tumor"
            OVERALL_SURVIVAL = "overall_survival"  # in days
            RESPONSE = "response"  # R, NR
            RECIST = "RECIST"  # PD, SD, PR, CR
            PRIOR_CTLA4 = "prior_CTLA4"
            PRIOR_PD1 = "prior_PD1"
            TREATMENT_CTLA4 = "treatment.CTLA4"
            TREATMENT_PD1 = "treatment.PD1"
            TREATMENT_PDL1 = "treatment.PDL1"
            TREATMENT_KIR = "treatment.KIR"
            TREATMENT_CD137 = "treatment.CD137"
            TREATMENT_IDO = "treatment.IDO"
            TREATMENT_LAG3 = "treatment.LAG-3"
            TREATMENT_ACT = "treatment.ACT"
            TREATMENT_CYTOSTASIS = "treatment.CYTOSTASIS"
            TIMING = "timing"
            SAMPLE_SOURCE = "sample_source"  # frozen or FFPE
            ORIGINAL_PATIENT_ID = "original.patient_id"
            ORIGINAL_SAMPLE_ID = "original.sample_id"

        def process_freeman(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["Age"]
            processed_clinical_data[Column.SEX.value] = clinical_data["Gender"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = clinical_data["Primary_tumor"].map({"Uveal": "UVM", "Skin": "SKCM", "Mucosal": "OTHER", "Penile": "OTHER", "Retinal": "OTHER", "Vulva": "OTHER"})
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["Overall_survival"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["Response "]
            processed_clinical_data[Column.RECIST.value] = np.nan
            processed_clinical_data[Column.PRIOR_CTLA4.value] = clinical_data.groupby('Patient', group_keys=False, sort=False).apply(partial(flag_seen, target_treatment="CTLA4")).explode().reset_index(drop=True).astype(bool)
            processed_clinical_data[Column.PRIOR_PD1.value] = clinical_data.groupby('Patient', group_keys=False, sort=False).apply(partial(flag_seen, target_treatment="PD1")).explode().reset_index(drop=True).astype(bool)
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = ((clinical_data["Therapy"] == "CTLA4") | (clinical_data["Therapy"] == "CTLA4+PD1"))
            processed_clinical_data[Column.TREATMENT_PD1.value] = ((clinical_data["Therapy"] == "PD1") | (clinical_data["Therapy"] == "CTLA4+PD1") | (clinical_data["Therapy"] == "PD1+KIR"))
            processed_clinical_data[Column.TREATMENT_PDL1.value] = (clinical_data["Therapy"] == "PDL1")
            processed_clinical_data[Column.TREATMENT_KIR.value] = (clinical_data["Therapy"] == "PD1+KIR")
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TIMING.value] = clinical_data["Timing"]
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = clinical_data["Source_of_sample"]
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["Patient"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["Sample_id"]
            return processed_clinical_data

        def process_snyder(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["Age"]
            processed_clinical_data[Column.SEX.value] = clinical_data["Sex"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "BLCA"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["OS in days"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["Best Response RECIST 1.1"].map({"SD": "NR", "PD": "NR", "PR": "R", "CR": "R", "Only scanned baseline": np.nan})
            processed_clinical_data[Column.RECIST.value] = clinical_data["Best Response RECIST 1.1"].map(lambda x: np.nan if re.fullmatch(r"Only scanned (at )?baseline", x) else x)
            processed_clinical_data[Column.PRIOR_CTLA4.value] = np.nan
            processed_clinical_data[Column.PRIOR_PD1.value] = np.nan
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = False
            processed_clinical_data[Column.TREATMENT_PD1.value] = False
            processed_clinical_data[Column.TREATMENT_PDL1.value] = True
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TIMING.value] = "Pre"
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["patient_id"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["sample_id_rna_tumor"]
            return processed_clinical_data

        def process_riaz(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = np.nan
            processed_clinical_data[Column.SEX.value] = np.nan
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "SKCM"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = 7*clinical_data["Time to Death\n(weeks)"].astype(float)
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["Response"].map({"SD": "NR", "PD": "NR", "PR": "R", "CR": "R"})
            processed_clinical_data[Column.RECIST.value] = clinical_data["Response"]
            processed_clinical_data[Column.PRIOR_CTLA4.value] = np.nan
            processed_clinical_data[Column.PRIOR_PD1.value] = np.nan
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = False
            processed_clinical_data[Column.TREATMENT_PD1.value] = True
            processed_clinical_data[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TIMING.value] = clinical_data["original.sample_id"].apply(lambda x: x.split("_")[1])
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["origional.patient_id"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["original.sample_id"]
            return processed_clinical_data

        def process_rose(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["age"]
            processed_clinical_data[Column.SEX.value] = clinical_data["gender"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = clinical_data["cancer_type"]
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["os (days)"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["response_label"]
            processed_clinical_data[Column.RECIST.value] = clinical_data["RECIST"]
            processed_clinical_data[Column.PRIOR_CTLA4.value] = np.nan
            processed_clinical_data[Column.PRIOR_PD1.value] = np.nan
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = (clinical_data["ICI_target"]=="CTLA-4")
            processed_clinical_data[Column.TREATMENT_PD1.value] = (clinical_data["ICI_target"]=="PD1")
            processed_clinical_data[Column.TREATMENT_PDL1.value] = (clinical_data["ICI_target"]=="PD-L1")
            processed_clinical_data[Column.TREATMENT_KIR.value] = (clinical_data["ICI_target"]=="KIR")
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TIMING.value] = clinical_data["Timing"]
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["patient_id"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["Sample_id"]
            return processed_clinical_data

        def process_ravi(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["Patient_Age_at_Diagnosis"].map(lambda x: {">89": "89"}.get(x, x)).astype(float)
            processed_clinical_data[Column.SEX.value] = clinical_data["Patient_Sex"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "LUAD"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["Harmonized_OS_Days"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["Harmonized_Confirmed_BOR"].map(recist_to_response_map)
            processed_clinical_data[Column.RECIST.value] = clinical_data["Harmonized_Confirmed_BOR"]
            processed_clinical_data[Column.PRIOR_CTLA4.value] = np.nan
            processed_clinical_data[Column.PRIOR_PD1.value] = np.nan
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = ["CTLA-4" in ici_agent_to_ici_target(agent) for agent in clinical_data["Agent_PD1"]]
            processed_clinical_data[Column.TREATMENT_PD1.value] = ["PD-1" in ici_agent_to_ici_target(agent) for agent in clinical_data["Agent_PD1"]]
            processed_clinical_data[Column.TREATMENT_PDL1.value] = ["PD-L1" in ici_agent_to_ici_target(agent) for agent in clinical_data["Agent_PD1"]]
            processed_clinical_data[Column.TREATMENT_KIR.value] = ["KIR" in ici_agent_to_ici_target(agent) for agent in clinical_data["Agent_PD1"]]
            processed_clinical_data[Column.TREATMENT_CD137.value] = ["CD137" in ici_agent_to_ici_target(agent) for agent in clinical_data["Agent_PD1"]]
            processed_clinical_data[Column.TREATMENT_IDO.value] = ["IDO" in ici_agent_to_ici_target(agent) for agent in clinical_data["Agent_PD1"]]
            processed_clinical_data[Column.TREATMENT_LAG3.value] = ["LAG-3" in ici_agent_to_ici_target(agent) for agent in clinical_data["Agent_PD1"]]
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = ["Cytostasis" in ici_agent_to_ici_target(agent) for agent in clinical_data["Agent_PD1"]]
            processed_clinical_data[Column.TIMING.value] = "Pre"
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["Harmonized_SU2C_Participant_ID_v2"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["Harmonized_SU2C_RNA_Tumor_Sample_ID_v2"]
            return processed_clinical_data

        def process_chen(clinical_data):

            # aCTLA-4
            processed_clinical_data_CTLA4 = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data_CTLA4[Column.AGE.value] = np.nan
            processed_clinical_data_CTLA4[Column.SEX.value] = np.nan
            processed_clinical_data_CTLA4[Column.PRIMARY_TUMOR.value] = "SKCM" # "Mostly metastatic melanoma" according to paper
            processed_clinical_data_CTLA4[Column.OVERALL_SURVIVAL.value] = np.nan
            processed_clinical_data_CTLA4[Column.RESPONSE.value] = clinical_data["anti-CTLA-4 response"]
            processed_clinical_data_CTLA4[Column.RECIST.value] = np.nan
            processed_clinical_data_CTLA4[Column.PRIOR_CTLA4.value] = False
            processed_clinical_data_CTLA4[Column.PRIOR_PD1.value] = False
            processed_clinical_data_CTLA4[Column.TREATMENT_CTLA4.value] = True
            processed_clinical_data_CTLA4[Column.TREATMENT_PD1.value] = False
            processed_clinical_data_CTLA4[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data_CTLA4[Column.TREATMENT_KIR.value] = False
            processed_clinical_data_CTLA4[Column.TREATMENT_CD137.value] = False
            processed_clinical_data_CTLA4[Column.TREATMENT_IDO.value] = False
            processed_clinical_data_CTLA4[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data_CTLA4[Column.TREATMENT_ACT.value] = False
            processed_clinical_data_CTLA4[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data_CTLA4[Column.TIMING.value]= clinical_data["Timepoint"].map({'On aPD1': "Post", 'Pre aPD1': "Post", 'Pre-aCTLA4': "Post", 'Prog aCTLA4 / Pre aPD1': "Post", 'Prog aPD1': "Post", 'on aCTLA4': "On"})
            processed_clinical_data_CTLA4[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data_CTLA4[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["Patient ID (int)"]
            processed_clinical_data_CTLA4[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["index"]

            # aPD1
            processed_clinical_data_PD1 = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data_PD1[Column.AGE.value] = np.nan
            processed_clinical_data_PD1[Column.SEX.value] = np.nan
            processed_clinical_data_PD1[Column.PRIMARY_TUMOR.value] = "SKCM" # "Mostly metastatic melanoma" according to paper
            processed_clinical_data_PD1[Column.OVERALL_SURVIVAL.value] = np.nan
            processed_clinical_data_PD1[Column.RESPONSE.value] = clinical_data["anti-PD-1 response"]
            processed_clinical_data_PD1[Column.RECIST.value] = np.nan
            processed_clinical_data_PD1[Column.PRIOR_CTLA4.value] = clinical_data["Timepoint"].map({'On aPD1': True, 'Pre aPD1': True, 'Pre-aCTLA4': False, 'Prog aCTLA4 / Pre aPD1': True, 'Prog aPD1': True, 'on aCTLA4': False})
            processed_clinical_data_PD1[Column.PRIOR_PD1.value] = False
            processed_clinical_data_PD1[Column.TREATMENT_CTLA4.value] = False
            processed_clinical_data_PD1[Column.TREATMENT_PD1.value] = True
            processed_clinical_data_PD1[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data_PD1[Column.TREATMENT_KIR.value] = False
            processed_clinical_data_PD1[Column.TREATMENT_CD137.value] = False
            processed_clinical_data_PD1[Column.TREATMENT_IDO.value] = False
            processed_clinical_data_PD1[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data_PD1[Column.TREATMENT_ACT.value] = False
            processed_clinical_data_PD1[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data_PD1[Column.TIMING.value] = clinical_data["Timepoint"].map({'On aPD1': "On", 'Pre aPD1': "Pre", 'Pre-aCTLA4': "Pre", 'Prog aCTLA4 / Pre aPD1': "Pre", 'Prog aPD1': "Post", 'on aCTLA4': "Pre"})
            processed_clinical_data_PD1[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data_PD1[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["Patient ID (int)"]
            processed_clinical_data_PD1[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["index"]

            return processed_clinical_data_CTLA4, processed_clinical_data_PD1

        def process_gide(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["Age (Years)"]
            processed_clinical_data[Column.SEX.value] = clinical_data["Sex"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "SKMC"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["Overall Survival (Days)"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["Best RECIST response"].apply(lambda x: recist_to_response_map[x.strip()])
            processed_clinical_data[Column.RECIST.value] = clinical_data["Best RECIST response"]
            processed_clinical_data[Column.PRIOR_CTLA4.value] = False
            processed_clinical_data[Column.PRIOR_PD1.value] = False
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = (clinical_data["original.sample_id"].apply(lambda x: x.split("_")[1]) == "ipiPD1")
            processed_clinical_data[Column.TREATMENT_PD1.value] = True
            processed_clinical_data[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TIMING.value] = clinical_data["original.sample_id"].apply(lambda x: x.split("_")[-1]).map({"PRE": "Pre", "EDT": "Post"})
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["original.patient_id"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["original.sample_id"]
            return processed_clinical_data

        def process_vanallen(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["age_start"]
            processed_clinical_data[Column.SEX.value] = clinical_data["gender"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "SKCM"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["overall_survival"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["RECIST"].map({**recist_to_response_map, "X": np.nan})
            processed_clinical_data[Column.RECIST.value] = clinical_data["RECIST"]
            processed_clinical_data[Column.PRIOR_CTLA4.value] = np.nan
            processed_clinical_data[Column.PRIOR_PD1.value] = np.nan
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = True
            processed_clinical_data[Column.TREATMENT_PD1.value] = False
            processed_clinical_data[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TIMING.value] = "Pre"
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["original.patient_id"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["original.sample_id"]
            return processed_clinical_data

        def process_lauss(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = np.nan
            processed_clinical_data[Column.SEX.value] = np.nan
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "SKMC"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["OS.Time"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["RECIST"].map({**recist_to_response_map, "X": np.nan})
            processed_clinical_data[Column.RECIST.value] = clinical_data["RECIST"]
            processed_clinical_data[Column.PRIOR_CTLA4.value] = np.nan # according to study many had previous CTLA-4 treatment
            processed_clinical_data[Column.PRIOR_PD1.value] = False
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = False
            processed_clinical_data[Column.TREATMENT_PD1.value] = False
            processed_clinical_data[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = True
            processed_clinical_data[Column.TIMING.value] = "Pre"
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["original.patient_id"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["original.sample_id"]
            return processed_clinical_data

        def process_auslander(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["Age"]
            processed_clinical_data[Column.SEX.value] = clinical_data["Sex"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "SKCM"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = np.nan
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["Response"]
            processed_clinical_data[Column.RECIST.value] = np.nan
            processed_clinical_data[Column.PRIOR_CTLA4.value] = np.nan
            processed_clinical_data[Column.PRIOR_PD1.value] = np.nan
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = ["anti-CTLA-4" in agent for agent in clinical_data["Treatment"]]
            processed_clinical_data[Column.TREATMENT_PD1.value] = ["anti-PD-1" in agent for agent in clinical_data["Treatment"]]
            processed_clinical_data[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TIMING.value] = "Pre"
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["Patient"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["original.sample_id"]
            return processed_clinical_data

        def process_liu(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = np.nan
            processed_clinical_data[Column.SEX.value] = np.nan
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "SKCM"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["Overall_survival"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["Response "]
            processed_clinical_data[Column.RECIST.value] = np.nan
            processed_clinical_data[Column.PRIOR_CTLA4.value] = clinical_data["Prior_CTLA4"]
            processed_clinical_data[Column.PRIOR_PD1.value] = False
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = False
            processed_clinical_data[Column.TREATMENT_PD1.value] = True
            processed_clinical_data[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TIMING.value] = clinical_data["Timing"]
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["original_sample_id"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["Patient"]
            return processed_clinical_data

        def process_hugo(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["Freeman.Age"]
            processed_clinical_data[Column.SEX.value] = clinical_data["Freeman.Gender"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = "SKCM"
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = clinical_data["Freeman.Overall_survival"]
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["Freeman.Response "]
            processed_clinical_data[Column.RECIST.value] = clinical_data["RECIST"]
            processed_clinical_data[Column.PRIOR_CTLA4.value] = clinical_data["Freeman.Prior_CTLA4"]
            processed_clinical_data[Column.PRIOR_PD1.value] = False
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = False
            processed_clinical_data[Column.TREATMENT_PD1.value] = True
            processed_clinical_data[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TIMING.value] = clinical_data["Timing"]
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = np.nan
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["Freeman.Patient"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["SRA"]
            return processed_clinical_data

        def process_prat(clinical_data):
            processed_clinical_data = pd.DataFrame(index=clinical_data.index)
            processed_clinical_data[Column.AGE.value] = clinical_data["AGE"]
            processed_clinical_data[Column.SEX.value] = clinical_data["SEX"]
            processed_clinical_data[Column.PRIMARY_TUMOR.value] = clinical_data["CANCER"].map({"SqH&N": "HNSC", "SKCM": "SKCM", "SqCLC": "LUSC", "Non-SqCLC": "LUAD"})
            processed_clinical_data[Column.OVERALL_SURVIVAL.value] = np.nan
            processed_clinical_data[Column.RESPONSE.value] = clinical_data["BEST.RESP"].map(recist_to_response_map)
            processed_clinical_data[Column.RECIST.value] = clinical_data["BEST.RESP"]
            processed_clinical_data[Column.PRIOR_CTLA4.value] = np.nan
            processed_clinical_data[Column.PRIOR_PD1.value] = np.nan
            processed_clinical_data[Column.TREATMENT_CTLA4.value] = False
            processed_clinical_data[Column.TREATMENT_PD1.value] = True
            processed_clinical_data[Column.TREATMENT_PDL1.value] = False
            processed_clinical_data[Column.TREATMENT_KIR.value] = False
            processed_clinical_data[Column.TREATMENT_CD137.value] = False
            processed_clinical_data[Column.TREATMENT_IDO.value] = False
            processed_clinical_data[Column.TREATMENT_LAG3.value] = False
            processed_clinical_data[Column.TREATMENT_CYTOSTASIS.value] = False
            processed_clinical_data[Column.TREATMENT_ACT.value] = False
            processed_clinical_data[Column.TIMING.value] = "Pre"
            processed_clinical_data[Column.SAMPLE_SOURCE.value] = clinical_data["BIOPSY"]
            processed_clinical_data[Column.ORIGINAL_PATIENT_ID.value] = clinical_data["original.patient_id"]
            processed_clinical_data[Column.ORIGINAL_SAMPLE_ID.value] = clinical_data["original.sample_id"]
            return processed_clinical_data

        output_dir = lair.get_path(self)

        ds = ImmuneCheckpointTherapyResponseProcessedGeneNormalized()
        lair.safe_derive(ds)
        filepaths = lair.get_dataset_filepaths(ds)

        adata = ad.read_h5ad(filepaths["Freeman.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_freeman(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Freeman.h5ad"))

        adata = ad.read_h5ad(filepaths["Snyder.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_snyder(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Snyder.h5ad"))

        adata = ad.read_h5ad(filepaths["Riaz.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_riaz(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Riaz.h5ad"))

        adata = ad.read_h5ad(filepaths["Rose.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_rose(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Rose.h5ad"))

        adata = ad.read_h5ad(filepaths["Ravi.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_ravi(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Ravi.h5ad"))

        adata = ad.read_h5ad(filepaths["Chen.h5ad"])
        chen_clinical_data_CTLA4, chen_clinical_data_PD1 = process_chen(adata.obs.reset_index(drop=True))
        adata.obs = normalize_values_of_clinical_data(chen_clinical_data_CTLA4)
        adata.write(output_dir.joinpath("Chen-CTLA4.h5ad"))
        adata.obs = normalize_values_of_clinical_data(chen_clinical_data_PD1)
        adata.write(output_dir.joinpath("Chen-PD1.h5ad"))

        adata = ad.read_h5ad(filepaths["Gide.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_gide(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Gide.h5ad"))

        adata = ad.read_h5ad(filepaths["VanAllen.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_vanallen(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("VanAllen.h5ad"))

        adata = ad.read_h5ad(filepaths["Lauss.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_lauss(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Lauss.h5ad"))

        adata = ad.read_h5ad(filepaths["Auslander.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_auslander(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Auslander.h5ad"))

        adata = ad.read_h5ad(filepaths["Liu.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_liu(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Liu.h5ad"))

        adata = ad.read_h5ad(filepaths["Hugo.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_hugo(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Hugo.h5ad"))

        adata = ad.read_h5ad(filepaths["Prat.h5ad"])
        adata.obs = normalize_values_of_clinical_data(process_prat(adata.obs.reset_index(drop=True)))
        adata.write(output_dir.joinpath("Prat.h5ad"))