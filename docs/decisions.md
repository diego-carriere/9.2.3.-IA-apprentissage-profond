# Décisions

Chaque entrée : décision, raison, date. Les plus récentes en bas.

## 2026-11-19 — Phase 0 (développement sur CPU, sans GPU)

1. **Développement sur CPU avec Qwen3-0.6B, cible Qwen3-4B-Instruct-2507 sur GPU.**
   Les deux modèles ont le même tokenizer, donc le même masque (vérifié dans `docs/reports/token_mask.md`). Tout le pipeline est testé ici ; sur GPU, seul le nom du modèle change.

2. **Environnement CPU en Python 3.14** (venv `../.venv`, `requirements.txt`). transformers 5.23 et sentence-transformers 6.3 s'y installent. L'environnement GPU (Python 3.12, CUDA ≥ 12.8, Unsloth, bitsandbytes) sera figé à part dans `requirements-gpu.txt`, sur le PC GPU.

3. **Définition du « e » interdit** (`rules.py`) : un caractère est interdit s'il appartient à `FORBIDDEN_CHARS` ou si sa décomposition NFKD en contient un. Cela couvre `e` + accent combinant, ẽ, ę, ainsi que ｅ (pleine chasse) et 𝐞 (mathématique). Au total, 97 caractères Unicode sont interdits. Les positions se calculent sur le texte NFC.

4. **Masque de tokens : règle exacte aux frontières UTF-8 au lieu de « bannir tout token incomplet ».** La règle large du prompt (puis une première version plus fine, qui bannissait tout octet de continuation pouvant terminer *un* caractère interdit) rendait « ù », « û », « ï », « ÿ », « Ù », « Û » et « Ÿ » impossibles à produire : Qwen les tokenise en `Ã` + `¹`, et `¹` termine aussi ẹ (`E1 BA B9`). La règle retenue bannit un seul côté de chaque coupure possible :
   - lead `C3`/`C5` (2 octets, partagés avec les lettres françaises) : on bannit le token qui *commence* par l'octet de continuation interdit (`A9` pour é) ;
   - tout autre lead : on bannit le token qui *finit* par un préfixe incomplet du caractère.
   Test exhaustif sur toutes les paires (fin incomplète × début continuation), environ 200 k pour Qwen, plus 100 k paires aléatoires. Un test de mutation vérifie que le test détecte bien une règle cassée. Résultat : 31,9 % du vocabulaire Qwen interdit.

5. **Tokens spéciaux : seuls EOS et fin de tour (`<|im_end|>`, `<|endoftext|>`) sont autorisés**, et non tous les tokens spéciaux comme le demande le prompt : `<think>`, les appels d'outils, etc. feraient dérailler la génération.

6. **`repetition_penalty` = 1.0 et `no_repeat_ngram_size` = 0 par défaut.** Dans HF, ces deux processeurs voient aussi le prompt : ils pénaliseraient la recopie des mots sans « e » de la source, donc la fidélité. Pour limiter les boucles, on plafonne `max_new_tokens` à 2 × longueur de la source + 32, puis on s'appuie sur la détection de dégénérescence et le best-of-n.

7. **Best-of-n évalué par préfixes** : on génère 16 candidats par texte une seule fois ; le best-of-n pour n se calcule sur les n premiers. Les candidats étant i.i.d., l'estimation est sans biais et coûte 4 fois moins cher que 4 passes.

8. **Embedding pour `sem_sim` : `intfloat/multilingual-e5-small`** (118 M paramètres, environ 470 Mo, tourne sur CPU dans la RAM disponible, multilingue). On utilise le préfixe `query: ` des deux côtés, comme recommandé pour une tâche symétrique. Limite connue : biais lexical. Un lipogramme change forcément de mots, ce qui fait baisser `sem_sim`. On compare donc des variantes entre elles, sans viser un seuil absolu.

9. **`len_ratio` en caractères.** Les lipogrammes emploient des mots de longueur différente ; les caractères sont moins biaisés que les mots.

10. **`fluency` = log-probabilité moyenne par token sous le modèle de base non masqué.** Biais structurel : un lipogramme emploie des mots rares. On n'interprète que des écarts entre variantes ; la notation humaine fait foi. Pour le reranking, on utilise `exp(fluency)`, la probabilité moyenne géométrique par token, déjà dans [0, 1], sans constante arbitraire.

11. **Score composite du reranking** : 0,5 × `sem_sim` + 0,3 × `exp(fluency)` + 0,2 × score de longueur (1 dans [0,7 ; 1,3], décroissance linéaire jusqu'à 0 à 0,5 au-delà). Poids de départ, à revoir après la notation humaine de l'étape 1.

12. **Inférence du 4B en 4-bit (nf4, bitsandbytes).** En bf16, le 4B pèse environ 8 Go et ne tient pas en VRAM. `fluency` est donc mesurée sous le modèle quantifié (impact à vérifier sur un échantillon).

13. **Approximation de la fréquence des tokens dans le rapport du masque : ordre des fusions BPE.** Elle est biaisée vers l'anglais. Elle sera remplacée par des fréquences sur le corpus français une fois celui-ci construit.

14. **Jeu d'évaluation figé : extrait de 19 livres français du domaine public (Project Gutenberg)** via `scripts/build_eval.py` (graine fixe). Le site principal de Gutenberg était en maintenance ; on passe par le miroir `mirrors.xmission.com`. Wikisource renvoyait des extraits vides à cause de la transclusion.
    - 200 phrases (40 par genre : narratif, descriptif, dialogue, argumentatif, technique), 50 paragraphes (10 par genre) et 30 pièges.
    - Tous les auteurs sont morts avant 1915. L'en-tête et la licence Gutenberg sont retirés ; seul le texte du domaine public est conservé.
    - Dialogue : Gutenberg note le tiret `--`, normalisé en `—`. Un paragraphe de dialogue = une suite d'au moins 2 répliques consécutives.
    - Pièges : score automatique (noms propres contenant « e », nombres en lettres, verbes en -ent, guillemets), au moins 2 critères, 3 par livre au maximum.
    - **Limite** : registre exclusivement XIXe siècle ou plus ancien (Descartes, Voltaire). Un jeu moderne complémentaire serait utile pour mesurer la généralisation.
    - **Hash** : `f45b381a316d8715` (sha256 de `data/eval/MANIFEST.sha256`, tronqué), vérifié par `tests/test_eval_set.py`.
    - Ces 19 livres (identifiants Gutenberg dans les champs `source.gutenberg_id`) sont **exclus du corpus d'entraînement**.

15. **Mode A autorisé** (confirmé par l'utilisateur le 2026-11-19) : l'enseignant `claude -p` (abonnement Claude Code) peut générer des données pour entraîner ce petit modèle de cours.

16. **`sem_sim` (e5) mal calibrée** : sur le smoke test, elle vaut environ 0,91 pour des sorties à peine intelligibles (les cosinus d'e5 sont tassés dans [0,7 ; 1]). À faire avant l'étape 1 : la recentrer par rapport à une référence de paires non liées (`(sim − sim_aléatoire) / (1 − sim_aléatoire)`), et ajouter un taux de mots hors lexique pour attraper les mots tronqués (« vall », « lumi »), invisibles pour `degenerate_rate`.

17. **Enseignant du mode A : Sonnet en `--effort low` via `claude -p`, protocole à objet JSON numéroté, lots de 10.**
    - Premier essai (tableau JSON, lots de 20) : 19 réponses sur 20, donc tout le lot perdu ; 42 k tokens de réflexion pour un seul appel de 6,5 min.
    - La réflexion ne peut pas être désactivée (ni `alwaysThinkingEnabled`, ni `MAX_THINKING_TOKENS`). `--effort low` donne une qualité équivalente à `medium`. Haiku fait pire : deux fois plus de réflexion, plus d'erreurs.
    - Les réponses manquantes sont redemandées au tour de réparation.
    - Aucune clé API : les variables `ANTHROPIC_API_KEY` et apparentées sont retirées de l'environnement du sous-processus, et `--setting-sources ""` est passé. Le `total_cost_usd` est une estimation au tarif public, non facturée.

18. **Contrôle lexique (pyspellchecker FR + `configs/lexicon_extra.txt`) dans la boucle de réparation et comme filtre.**
    - Sur les 50 premières paires sans ce contrôle, 21 sur 46 contenaient des mots inventés (« musicos », « hannutons », « chadival ») et 34 sur 46 commençaient par une minuscule. Entraîner dessus aurait appris à l'élève la pathologie même qu'on veut corriger.
    - Avec le contrôle : 0 mot inventé, 0 minuscule en tête, 43 paires gardées sur 50 (86 %), environ 0,3 appel par source.
    - Les mots à apostrophe ou trait d'union sont testés en entier, puis par parties. Les clitiques sont acceptés (« dit-on », « va-t-il »).
    - La liste complémentaire ne contient que des mots réels vérifiés (« frangin », « toubib »…).
    - Nouvelle métrique d'évaluation : `oov_rate`.

19. **Le seuil `min_sem_sim` = 0,85 reste provisoire.** Sur 50 paires, `sem_sim` va de 0,88 à 0,95 ; elle ne sépare ni les contresens ni les mots inventés. Elle sera recalibrée (recentrage par rapport à des paires non liées) avant l'étape 1.

