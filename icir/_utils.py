import pandas as pd
import anndata as ad
import numpy as np
import pyensembl
import re
import mygene

def normalize_gene(id, ensembl, original_id=None):
    if id is None:
        return pd.Series({"original.id": id})
    try:
        if original_id is None:
            original_id = id
        gene = ensembl.gene_by_id(id.split(".")[0])
        gene_normalized = pd.Series(gene.to_dict())
        gene_normalized["EnsemblRelease.release"] = gene_normalized["genome"].release
        gene_normalized["EnsemblRelease.species"] = gene_normalized["genome"].species.latin_name
        gene_normalized = gene_normalized.drop("genome")
        gene_normalized["original.id"] = original_id
        return gene_normalized
    except:
        return pd.Series({"original.id": id})

def detect_pre_id_transform(var_names, ensembl):
    if np.all([bool(re.fullmatch(r"ENSG\d{11}", name)) for name in var_names]):
        pre_id_transform = None
    elif np.all([bool(re.fullmatch(r"ENST\d{11}", name)) for name in var_names]):
        pre_id_transform = "transcript"
    elif np.all([bool(re.fullmatch(r"\d{1,10}", name)) for name in var_names]):
        pre_id_transform = "entrez"
    elif sum(name in ensembl.gene_names() for name in var_names) >= 0.1 * len(var_names):
        pre_id_transform = "hugo"
    else:
        raise ValueError("Could not automatically determine pre_id_transform!")
    return pre_id_transform

def normalize_all_genes(var_names, pre_id_transform=None):
    ensembl = pyensembl.EnsemblRelease(release=111, species="human")
    original_ids = list(var_names)

    if pre_id_transform == "auto":
        pre_id_transform = detect_pre_id_transform(var_names, ensembl)

    match pre_id_transform:
        case "transcript":
            ids = list()
            for id in list(var_names):
                try:
                    gene_id = ensembl.transcript_by_id(id).gene_id
                    ids.append(gene_id)
                except ValueError:
                    ids.append(None)
        case "entrez":
            result = mygene.MyGeneInfo().getgenes(list(var_names), fields="symbol,name,ensembl")
            ids = list()
            for gene in result:
                if "ensembl" not in gene.keys():
                    ids.append(None)
                    continue
                if type(gene["ensembl"]) is list:
                    ids.append(None)
                    continue
                ids.append(gene["ensembl"]["gene"])
        case "hugo":
            ids = []
            for gene_name in var_names:
                try:
                    ensembl_ids = ensembl.gene_ids_of_gene_name(gene_name)
                except ValueError:
                    ids.append(None)
                    continue
                if len(ensembl_ids) == 1:
                    ids.append(ensembl_ids[0])
                else:
                    ids.append(None)
        case None:
            ids = list(var_names)
    genes = pd.concat([normalize_gene(gene_id, ensembl, original_id) for gene_id, original_id in zip(ids, original_ids)], axis=1).T
    return genes
