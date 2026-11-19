def calculate(severity, traffic, importance, repeats):
    score=0.40*severity+0.25*traffic+0.20*importance+0.15*repeats
    return round(score,1), ('P1 — Urgent' if score>=75 else 'P2 — High' if score>=50 else 'P3 — Normal')
