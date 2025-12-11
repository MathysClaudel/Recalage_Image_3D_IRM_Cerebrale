import os
import glob
import subprocess
import time
import shutil
import sys
import argparse

"""
Script d'automatisation UNIFIÉ pour featMatchMultiple.
Fonctionnalités :
1. Logique robuste & Clean (aligned with v2 logic)
2. Gestion de l'option -r- via argument --no_rotation
3. Anti Auto-Match
4. Nettoyage agressif
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Générateur de matches SIFT Unifié.")
    parser.add_argument("--patients", required=True, help="Dossier contenant les fichiers .key des images CIBLES")
    parser.add_argument("--atlases", required=True, help="Dossier contenant les fichiers .key des images SOURCES")
    parser.add_argument("--output", required=True, help="Dossier où seront stockés les résultats")
    parser.add_argument("--exe", required=True, help="Chemin vers l'exécutable featMatchMultiple")
    parser.add_argument("--no_rotation", action="store_true", help="Si activé, ajoute l'option -r- (Désactive la rotation)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    # 1. Chemins absolus
    exe_path = os.path.abspath(args.exe)
    patients_dir = os.path.abspath(args.patients)
    atlases_dir = os.path.abspath(args.atlases)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(exe_path):
        print(f"[ERREUR] Exécutable introuvable : {exe_path}")
        sys.exit(1)
    
    targets = sorted(glob.glob(os.path.join(patients_dir, "*.key")))
    sources = sorted(glob.glob(os.path.join(atlases_dir, "*.key")))

    if not targets or not sources:
        print("[ERREUR] Aucun fichier .key trouvé dans les dossiers spécifiés.")
        sys.exit(1)

    print(f"--- DÉMARRAGE DU GÉNÉRATEUR ---")
    if args.no_rotation:
        print("!!! MODE : Rotation DÉSACTIVÉE (Option -r- utilisée) !!!")
    else:
        print("--- MODE : Rotation ACTIVÉE (Comportement par défaut) ---")

    print(f"Cibles : {len(targets)}")
    print(f"Sources: {len(sources)}")
    
    start_time_global = time.time()

    # BOUCLE PRINCIPALE
    for i, path_target in enumerate(targets):
        target_filename = os.path.basename(path_target)
        # Extraction ID propre
        target_id = target_filename.split('_')[0].split('.')[0] 
        if target_filename.startswith("sub-"):
             target_id = target_filename.split('_')[0]
        
        target_output_dir = os.path.join(output_dir, target_id)
        if not os.path.exists(target_output_dir):
            os.makedirs(target_output_dir)
        
        print(f"\n[{i+1}/{len(targets)}] Cible : {target_id}")
        
        for j, path_source in enumerate(sources):
            source_filename = os.path.basename(path_source)
            raw_source_id = source_filename.split('_')[0].replace(".key", "")
            final_source_id = f"sub-{raw_source_id}" if not raw_source_id.startswith("sub-") else raw_source_id

            # Fichiers finaux attendus
            dest_match1 = os.path.join(target_output_dir, f"match_{final_source_id}.img1.txt")
            dest_match2 = os.path.join(target_output_dir, f"match_{final_source_id}.img2.txt")

            # --- FIX ANTI AUTO-MATCH ---
            t_clean = target_id.replace("sub-", "")
            s_clean = final_source_id.replace("sub-", "")

            if t_clean == s_clean:
                if os.path.exists(dest_match1): 
                    try: os.remove(dest_match1)
                    except: pass
                if os.path.exists(dest_match2): 
                    try: os.remove(dest_match2)
                    except: pass
                continue

            if os.path.exists(dest_match1) and os.path.exists(dest_match2):
                continue

            # --- EXÉCUTION (GÉRÉE PAR L'ARGUMENT) ---
            if args.no_rotation:
                # Mode Sans Rotation (-r-)
                cmd = [exe_path, "-r-", path_target, path_source]
            else:
                # Mode Défaut (Avec Rotation)
                cmd = [exe_path, path_target, path_source]
            
            try:
                subprocess.run(cmd, cwd=target_output_dir, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Récupération résultats
                motif_recherche = os.path.join(atlases_dir, f"{raw_source_id}*.matches.img1.txt")
                candidats = glob.glob(motif_recherche)
                
                moved = False
                if candidats:
                    f_src_1 = candidats[0]
                    f_src_2 = f_src_1.replace(".img1.txt", ".img2.txt")
                    
                    shutil.move(f_src_1, dest_match1)
                    if os.path.exists(f_src_2):
                        shutil.move(f_src_2, dest_match2)
                    moved = True
                
                if moved:
                    sys.stdout.write(f"\r   -> vs {final_source_id} : OK   ")
                else:
                    sys.stdout.write(f"\r   -> vs {final_source_id} : - (Non trouvé) ")
                sys.stdout.flush()

            except Exception as e:
                print(f" [Erreur: {e}]")

            # --- NETTOYAGE AGRESSIF ---
            extensions_poubelle = [
                ".matches.info.txt",
                ".trans.txt",
                ".trans-inverse.txt",
                ".update.key"
            ]
            dossiers_a_nettoyer = [target_output_dir, atlases_dir, patients_dir]

            for dossier in dossiers_a_nettoyer:
                for ext in extensions_poubelle:
                    fichiers_sales = glob.glob(os.path.join(dossier, f"*{ext}"))
                    for f_sale in fichiers_sales:
                        try: os.remove(f_sale)
                        except OSError: pass

    print(f"\n\n--- TERMINÉ en {(time.time() - start_time_global)/60:.1f} minutes ---")

if __name__ == "__main__":
    main()