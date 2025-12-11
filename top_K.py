import numpy as np
import os
import glob
import argparse
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SUFFIXE_GT_BASE = "_space-T1w_desc-groundtruth_afids"

# --- FONCTIONS ROBUSTES ---
def charger_fichier_matches_robuste(path):
    try:
        data = []
        with open(path, 'r') as f: lines = f.readlines()
        offset = -1
        for line in lines:
            if line.startswith('#'): continue
            parts = line.split()
            # Recherche heuristique du début des données numériques
            for i in range(1, len(parts)-2):
                try:
                    if float(parts[i]) and float(parts[i+1]) and float(parts[i+2]):
                        offset = i; break
                except: continue
            if offset != -1: break
        if offset == -1: return None 
        for line in lines:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) < offset + 3: continue
            try: data.append([float(parts[offset]), float(parts[offset+1]), float(parts[offset+2])])
            except: continue
        return np.array(data)
    except: return None

def trouver_gt(dossier, nom_sujet):
    # Nettoyage
    nom_clean = nom_sujet.replace("_T1w", "").replace(".nii", "").replace(".gz", "")
    noms = [nom_clean]
    if nom_clean.startswith("sub-"): noms.append(nom_clean.replace("sub-", ""))
    else: noms.append(f"sub-{nom_clean}")
    
    for n in noms:
        # Priorité .fcsv
        p1 = os.path.join(dossier, f"{n}{SUFFIXE_GT_BASE}.fcsv")
        if os.path.exists(p1): return np.loadtxt(p1, delimiter=',', comments='#', usecols=(1, 2, 3))
        # Fallback .csv
        p2 = os.path.join(dossier, f"{n}{SUFFIXE_GT_BASE}.csv")
        if os.path.exists(p2): return np.loadtxt(p2, delimiter=',', comments='#', usecols=(1, 2, 3))
        # Fallback court
        p3 = os.path.join(dossier, f"{n}.fcsv")
        if os.path.exists(p3): return np.loadtxt(p3, delimiter=',', comments='#', usecols=(1, 2, 3))
    return None

def calculer_affine_ransac(pts_src, pts_dst, min_samples=5, residual_threshold=15.0):
    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold, random_state=42)
    try:
        ransac.fit(pts_src, pts_dst)
        nb_inliers = np.sum(ransac.inlier_mask_)
        if nb_inliers < min_samples: return None, 0
        src_in = pts_src[ransac.inlier_mask_]; dst_in = pts_dst[ransac.inlier_mask_]
        col_un = np.ones((len(src_in), 1)); A = np.hstack([src_in, col_un])
        solution, _, _, _ = np.linalg.lstsq(A, dst_in, rcond=None)
        return solution.T, nb_inliers
    except: return None, 0

# --- CŒUR DU CALCUL ---
def analyser_patient_tous_k(nom_dossier_cible, args):
    # 1. GT Cible
    gt_cible = trouver_gt(args.gt_target, nom_dossier_cible)
    if gt_cible is None: return None

    # 2. Matches
    path_matches = os.path.join(args.results, nom_dossier_cible)
    fichiers = glob.glob(os.path.join(path_matches, "match_*.img1.txt"))
    if not fichiers: return None

    candidats = [] 
    
    for f_img1 in fichiers:
        f_img2 = f_img1.replace(".img1.txt", ".img2.txt")
        if not os.path.exists(f_img2): continue

        base = os.path.basename(f_img1)
        # Ex: match_sub-0086_T1w.img1.txt -> sub-0086
        nom_source_raw = base.replace("match_", "").replace(".img1.txt", "").replace("_T1w", "")
        
        gt_source = trouver_gt(args.gt_source, nom_source_raw)
        if gt_source is None: continue

        p_a = charger_fichier_matches_robuste(f_img1)
        p_b = charger_fichier_matches_robuste(f_img2)
        if p_a is None or p_b is None or len(p_a) < args.min_samples: continue

        try:
            # Cas 2 (p_b -> p_a) : On calcule la transfo
            mat, score = calculer_affine_ransac(p_b, p_a, min_samples=args.min_samples, residual_threshold=args.threshold)
            
            if mat is not None:
                # On projette les landmarks de l'atlas vers le patient
                lm_homog = np.hstack([gt_source, np.ones((len(gt_source), 1))])
                pred = np.dot(lm_homog, mat.T)
                candidats.append((score, pred))
        except: continue

    if not candidats: return None

    # 3. Calcul progressif K
    candidats.sort(key=lambda x: x[0], reverse=True)
    preds_triees = [c[1] for c in candidats]
    
    erreurs_par_k = []
    max_k = min(len(preds_triees), 30)
    
    for k in range(1, max_k + 1):
        subset = np.array(preds_triees[:k])
        # Médiane robuste
        pred_mediane = np.median(subset, axis=0)
        
        diff = pred_mediane - gt_cible
        tre = np.mean(np.sqrt(np.sum(diff**2, axis=1)))
        erreurs_par_k.append(tre)
        
    return erreurs_par_k

def main():
    parser = argparse.ArgumentParser(description="Analyse K Influence")
    
    # Arguments Obligatoires
    parser.add_argument("results", help="Dossier contenant les résultats")
    parser.add_argument("gt_target", help="Dossier GT des CIBLES (Patients)")
    parser.add_argument("gt_source", help="Dossier GT des SOURCES (Atlas)")
    
    # Arguments Sortie 
    parser.add_argument("--output_dir", default=".", help="Dossier de sauvegarde (défaut: courant)")
    parser.add_argument("--name_png", default="courbe_K_test.png", help="Nom du fichier PNG")
    parser.add_argument("--name_csv", default="stats_K.csv", help="Nom du fichier CSV")

    # Arguments Paramètres
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=12)
    
    args = parser.parse_args()

    print("--- ANALYSE K (Custom Output) ---")
    if not os.path.exists(args.results): 
        print("Erreur: Dossier résultats introuvable"); return
    
    # Création dossier sortie si besoin
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    patients = sorted([d for d in os.listdir(args.results) if os.path.isdir(os.path.join(args.results, d))])
    donnees_globales = {k: [] for k in range(1, 31)}
    
    print(f"Calcul sur {len(patients)} dossiers...")
    
    count_ok = 0
    for i, p in enumerate(patients):
        print(f"[{i+1}] {p}...", end="", flush=True)
        res = analyser_patient_tous_k(p, args)
        if res:
            for k_idx, err in enumerate(res):
                donnees_globales[k_idx + 1].append(err)
            print(" OK")
            count_ok += 1
        else:
            print(" ÉCHEC")

    if count_ok == 0:
        print("\nAucun patient analysé avec succès.")
        return

    # --- GRAPHIQUE & CSV ---
    k_axis, means, stds = [], [], []
    for k in range(1, 31):
        vals = donnees_globales[k]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            k_axis.append(k)
    
    if not means: return

    # Chemins complets
    path_csv = os.path.join(args.output_dir, args.name_csv)
    path_png = os.path.join(args.output_dir, args.name_png)

    # Sauvegarde CSV
    np.savetxt(path_csv, np.column_stack((k_axis, means, stds)), 
               header="K,Mean,Std", delimiter=",", fmt=["%d", "%.4f", "%.4f"])
    print(f"\nDonnées sauvegardées : {path_csv}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_axis, means, marker='o', color='#e67e22', label='Erreur Moyenne')
    plt.fill_between(k_axis, np.array(means)-np.array(stds), np.array(means)+np.array(stds), color='#e67e22', alpha=0.1, label='Écart-type')

    
    # 1. On trouve l'index du minimum
    min_err = min(means)
    index_best = means.index(min_err)

    # 2. On récupère le K et l'écart-type correspondants
    best_k = k_axis[index_best]
    best_std = stds[index_best] 

    # Partie graphique
    plt.plot(best_k, min_err, marker='*', color='gold', markersize=15, markeredgecolor='black', 
                label=f'Min: {min_err:.2f}mm (std: {best_std:.2f}) (K={best_k})')

    plt.title(f"Influence de K") 
    plt.xlabel("K (Nombre d'atlas utilisés)")
    plt.ylabel("Erreur Moyenne (mm)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(path_png)
    print(f"Graphique généré : {path_png}")

    # 3. PRINT DANS LE TERMINAL
    print("\n" + "="*40)
    print(f"MEILLEUR RÉSULTAT OBTENU (K={best_k}) :")
    print(f"Moyenne    : {min_err:.4f} mm")
    print(f"Écart-type : {best_std:.4f} mm") 
    print("="*40 + "\n")

if __name__ == "__main__":
    main()