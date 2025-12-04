import os
import glob
import subprocess
import numpy as np
import argparse
import shutil
import sys
import time
from sklearn.linear_model import RANSACRegressor

"""
Script de PRÉDICTION DE LANDMARKS (Production).
Correction : check=False pour tolérer les crashs de featMatchMultiple en fin d'exécution.
"""

# --- CONFIGURATION ---
SUFFIXE_GT_ATLAS = "_space-T1w_desc-groundtruth_afids"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Prédicteur de Landmarks SIFT-RANSAC (Multi-Atlas).")
    
    # Entrées / Sorties
    parser.add_argument("--input", required=True, help="Fichier .key ou dossier contenant les .key des patients à prédire")
    parser.add_argument("--atlas_dir", required=True, help="Dossier contenant les atlas (.key et leurs GT .fcsv)")
    parser.add_argument("--output", required=True, help="Dossier de sortie des prédictions (.fcsv)")
    parser.add_argument("--exe", required=True, help="Chemin vers l'exécutable featMatchMultiple")
    
    # Paramètres Algorithme
    parser.add_argument("--k", type=int, default=12, help="Nombre d'atlas à utiliser pour la fusion (K)")
    parser.add_argument("--min_samples", type=int, default=5, help="RANSAC: Min samples")
    parser.add_argument("--threshold", type=float, default=15.0, help="RANSAC: Residual threshold")
    parser.add_argument("--no_rotation", action="store_true", help="Si activé, ajoute -r- (Désactive rotation)")
    
    return parser.parse_args()

# --- 1. FONCTIONS FICHIERS ---

def charger_fichier_matches_robuste(path):
    try:
        data = []
        with open(path, 'r') as f: lines = f.readlines()
        offset = -1
        for line in lines:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 5: continue
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

def load_gt_atlas(atlas_dir, atlas_id):
    # Essai 1
    p1 = os.path.join(atlas_dir, f"{atlas_id}{SUFFIXE_GT_ATLAS}.fcsv")
    if os.path.exists(p1): return np.loadtxt(p1, delimiter=',', comments='#', usecols=(1, 2, 3))
    # Essai 2 (avec sub-)
    if not atlas_id.startswith("sub-"):
        p2 = os.path.join(atlas_dir, f"sub-{atlas_id}{SUFFIXE_GT_ATLAS}.fcsv")
        if os.path.exists(p2): return np.loadtxt(p2, delimiter=',', comments='#', usecols=(1, 2, 3))
    # Essai 3 (sans sub-)
    if atlas_id.startswith("sub-"):
        p3 = os.path.join(atlas_dir, f"{atlas_id.replace('sub-', '')}{SUFFIXE_GT_ATLAS}.fcsv")
        if os.path.exists(p3): return np.loadtxt(p3, delimiter=',', comments='#', usecols=(1, 2, 3))
    return None

def save_fcsv(output_path, points):
    with open(output_path, 'w') as f:
        f.write("# Markups fiducial file version = 4.13\n")
        f.write("# CoordinateSystem = LPS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, p in enumerate(points):
            label = i + 1
            f.write(f"vtkMRMLMarkupsFiducialNode_{i},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},0,0,0,1,1,1,0,{label},,\n")

# --- 2. FONCTIONS MATHS ---

def calculer_affine_ransac(pts_src, pts_dst, min_samples=5, threshold=15.0):
    if len(pts_src) < min_samples or len(pts_dst) < min_samples:
        return None, 0
    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=threshold, random_state=42)
    try:
        ransac.fit(pts_src, pts_dst)
        nb_inliers = np.sum(ransac.inlier_mask_)
        if nb_inliers < min_samples: return None, 0
        
        # Refinement
        src_in = pts_src[ransac.inlier_mask_]
        dst_in = pts_dst[ransac.inlier_mask_]
        col_un = np.ones((len(src_in), 1))
        A = np.hstack([src_in, col_un])
        solution, _, _, _ = np.linalg.lstsq(A, dst_in, rcond=None)
        return solution.T, nb_inliers
    except: return None, 0

# --- 3. TRAITEMENT ---

def predict_single_patient(patient_key_path, args, temp_root):
    # args contient maintenant des CHEMINS ABSOLUS (Garanti par le main)
    
    patient_filename = os.path.basename(patient_key_path)
    patient_id = patient_filename.split('_')[0].split('.')[0]
    if patient_filename.startswith("sub-"): patient_id = patient_filename.split('_')[0]

    work_dir = os.path.join(temp_root, patient_id)
    if os.path.exists(work_dir): shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    print(f"\n--- Traitement Patient : {patient_id} ---")
    
    atlas_keys = sorted(glob.glob(os.path.join(args.atlas_dir, "*.key")))
    candidates = []

    sys.stdout.write("  Matches en cours : ")
    
    for idx_atlas, atlas_key_path in enumerate(atlas_keys):
        atlas_filename = os.path.basename(atlas_key_path)
        raw_atlas_id = atlas_filename.split('_')[0].replace(".key", "")
        atlas_id = f"sub-{raw_atlas_id}" if not raw_atlas_id.startswith("sub-") else raw_atlas_id
        
        # Anti Auto-Match
        p_clean = patient_id.replace("sub-", "")
        a_clean = atlas_id.replace("sub-", "")
        if p_clean == a_clean: continue

        # --- A. GÉNÉRATION MATCHES ---
        cmd = [args.exe]
        if args.no_rotation: cmd.append("-r-")
        cmd.append(patient_key_path) 
        cmd.append(atlas_key_path)   

        match_file_1 = os.path.join(work_dir, f"match_{atlas_id}.img1.txt")
        match_file_2 = os.path.join(work_dir, f"match_{atlas_id}.img2.txt")
        
        try:
            # MODIFICATION ICI : check=False pour tolérer les crashs en fin de process
            subprocess.run(cmd, cwd=work_dir, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Récupération
            motif = os.path.join(args.atlas_dir, f"{raw_atlas_id}*.matches.img1.txt")
            trouves = glob.glob(motif)
            
            if trouves:
                f_src_1 = trouves[0]
                f_src_2 = f_src_1.replace(".img1.txt", ".img2.txt")
                shutil.move(f_src_1, match_file_1)
                if os.path.exists(f_src_2): shutil.move(f_src_2, match_file_2)
            else:
                sys.stdout.write("x") # Pas trouvé (Vraiment échoué)
                sys.stdout.flush()
                continue

        except Exception as e:
            # Erreur Python (ex: Fichier introuvable)
            print(f"\n[ERREUR PYTHON] : {e}")
            sys.stdout.flush()
            continue

        # --- B. CALCUL TRANSFORM ---
        p_patient = charger_fichier_matches_robuste(match_file_1)
        p_atlas = charger_fichier_matches_robuste(match_file_2)

        if p_patient is not None and p_atlas is not None and len(p_patient) >= args.min_samples:
            mat, n_inliers = calculer_affine_ransac(p_atlas, p_patient, 
                                                  min_samples=args.min_samples, 
                                                  threshold=args.threshold)
            
            if mat is not None:
                gt_atlas = load_gt_atlas(args.atlas_dir, raw_atlas_id)
                if gt_atlas is not None:
                    ones = np.ones((len(gt_atlas), 1))
                    gt_homog = np.hstack([gt_atlas, ones])
                    pred_landmarks = np.dot(gt_homog, mat.T)
                    candidates.append((n_inliers, pred_landmarks))
                    sys.stdout.write(".") 
                else:
                    sys.stdout.write("G") # Pas de GT
            else:
                sys.stdout.write("-") # Ransac fail
        else:
            sys.stdout.write("o") # Pas assez de points
        
        sys.stdout.flush()
        
        # Cleanup fichiers temp
        for ext in [".matches.info.txt", ".trans.txt", ".trans-inverse.txt", ".update.key"]:
            for f in glob.glob(os.path.join(work_dir, f"*{ext}")):
                try: os.remove(f)
                except: pass
            for f in glob.glob(os.path.join(args.atlas_dir, f"*{ext}")):
                try: os.remove(f)
                except: pass

    print("") 

    # --- 4. FUSION ---
    if not candidates:
        print("  [ERREUR] Aucune prédiction valide générée.")
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    k_final = min(len(candidates), args.k)
    top_subset = [c[1] for c in candidates[:k_final]]
    consensus_pred = np.median(np.array(top_subset), axis=0)
    
    output_fcsv = os.path.join(args.output, f"{patient_id}_predicted.fcsv")
    save_fcsv(output_fcsv, consensus_pred)
    
    print(f"  -> Fusion sur {k_final} atlas (Inliers Max: {candidates[0][0]}).")
    print(f"  -> Sauvegardé : {output_fcsv}")
    
    try: shutil.rmtree(work_dir)
    except: pass


def main():
    args = parse_arguments()
    
    # --- CORRECTION CRITIQUE : CONVERSION ABSOLUE DES CHEMINS ---
    args.exe = os.path.abspath(args.exe)
    args.input = os.path.abspath(args.input)
    args.atlas_dir = os.path.abspath(args.atlas_dir)
    args.output = os.path.abspath(args.output)
    
    # Vérifications
    if not os.path.exists(args.exe):
        print(f"[ERREUR FATALE] Executable introuvable : {args.exe}")
        sys.exit(1)
        
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # Liste patients
    if os.path.isfile(args.input):
        patients = [args.input]
    else:
        patients = sorted(glob.glob(os.path.join(args.input, "*.key")))
    
    if not patients:
        print("[ERREUR] Aucun fichier .key patient trouvé.")
        sys.exit(1)
        
    # Dossier Temp global
    temp_root = os.path.join(args.output, "temp_prediction_workspace")
    if not os.path.exists(temp_root): os.makedirs(temp_root)

    print(f"=== PRÉDICTION LANDMARKS (K={args.k}) ===")
    print(f"Patients : {len(patients)}")
    print(f"Atlas Dir: {args.atlas_dir}")
    if args.no_rotation: print("Option   : Rotation DÉSACTIVÉE")
    
    t0 = time.time()
    
    for p in patients:
        predict_single_patient(p, args, temp_root)
        
    # Nettoyage global
    try: shutil.rmtree(temp_root)
    except: pass
    
    print(f"\nTemps Total : {(time.time() - t0)/60:.1f} minutes.")

if __name__ == "__main__":
    main()