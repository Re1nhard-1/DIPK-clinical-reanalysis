from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
P = ROOT.parents[1]


def main():
    train = pd.read_csv(ROOT / 'train_input_identity.csv', dtype=str)[['GSM', 'source_label']]
    a = pd.read_csv(P / 'GSE25055_metadata.csv', dtype=str)[['GSM', 'pathologic_response_pcr_rd', 'dlda30_prediction']]
    a['series'] = 'GSE25055'
    b = pd.read_csv(P / 'method_screen/DIPK/inference_feasibility/GSE25065_candidate_metadata.csv', dtype=str)[['GSM', 'pathologic_response_pcr_rd']]
    b['series'] = 'GSE25065'
    j = train.merge(pd.concat([a, b], ignore_index=True), on='GSM', how='left', validate='one_to_one')
    j['matches_pathology'] = j.source_label == j.pathologic_response_pcr_rd
    j['matches_dlda30'] = j.source_label == j.dlda30_prediction
    j.to_csv(ROOT / 'training_label_check.csv', index=False)
    print({'records': len(j), 'matches_pathology': int(j.matches_pathology.sum()), 'matches_dlda30': int(j.matches_dlda30.sum())})


if __name__ == '__main__':
    main()
