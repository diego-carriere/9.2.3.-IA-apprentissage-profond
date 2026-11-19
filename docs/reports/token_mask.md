# Rapport — masque de tokens

Généré par `python scripts/mask_report.py`.

Masques Qwen3-4B-Instruct-2507 et Qwen3-0.6B identiques : **oui** (ce qui justifie le développement sur CPU avec le 0.6B).

## `Qwen/Qwen3-4B-Instruct-2507`

- Taille du vocabulaire : 151669 ; tokens interdits : **48351** (31.9%)
- Tokens spéciaux autorisés : ['<|endoftext|>', '<|im_end|>']
- Lettres `àâçîïôùûüÿÀÂÇÎÏÔÙÛÜŸ` représentables (seules et précédées d'un espace) : **oui**

50 tokens autorisés les plus courts (mots ≥ 2 lettres) : 'in', 'on', 'at', 'st', 'or', 'it', 'an', 'ar', 'al', 'ou', 'is', 'ic', 'ro', 'as', 'ct', 'nd', 'id', 'am', 'om', 'im', 'il', 'ur', 'ad', 'ch', 'ut', 'if', 'ol', 'th', 'ig', 'iv', 'od', 'ag', 'ay', 'ot', 'us', 'un', 'ul', 'ow', 'ab', 'um', 'tr', 'ck', 'lo', 'ap', 'ub', 'ly', 'av', 'ir', 'nt', 'pt'

50 tokens autorisés les plus fréquents (approximation : ordre des fusions BPE) : 'in', 'on', 'at', 'st', 'or', ' th', 'it', 'an', 'ar', 'al', 'ou', 'is', 'ing', 'ion', 'ic', 'ro', 'as', 'ct', 'nd', ' in', 'id', 'am', ' to', ' of', 'om', 'im', 'il', ' and', 'ur', 'ad', 'ch', 'ut', 'if', 'ol', 'th', 'ig', 'iv', 'od', 'ag', 'ay', 'ot', 'us', ' st', 'un', 'ul', 'ow', 'ation', ' for', 'ab', 'ort'

| Mot outil | Tokens | Tous autorisés |
|---|---|---|
| `'un'` | ['un'] | oui |
| `' un'` | ['Ġun'] | oui |
| `'au'` | ['au'] | oui |
| `' au'` | ['Ġau'] | oui |
| `'du'` | ['du'] | oui |
| `' du'` | ['Ġdu'] | oui |
| `'ma'` | ['ma'] | oui |
| `' ma'` | ['Ġma'] | oui |
| `'mon'` | ['mon'] | oui |
| `' mon'` | ['Ġmon'] | oui |
| `'son'` | ['son'] | oui |
| `' son'` | ['Ġson'] | oui |
| `'nos'` | ['nos'] | oui |
| `' nos'` | ['Ġnos'] | oui |
| `'vos'` | ['vos'] | oui |
| `' vos'` | ['Ġvos'] | oui |
| `'ou'` | ['ou'] | oui |
| `' ou'` | ['Ġou'] | oui |
| `'mais'` | ['ma', 'is'] | oui |
| `' mais'` | ['Ġmais'] | oui |
| `'donc'` | ['don', 'c'] | oui |
| `' donc'` | ['Ġdonc'] | oui |
| `'car'` | ['car'] | oui |
| `' car'` | ['Ġcar'] | oui |
| `'ainsi'` | ['ains', 'i'] | oui |
| `' ainsi'` | ['Ġainsi'] | oui |
| `'dans'` | ['d', 'ans'] | oui |
| `' dans'` | ['Ġdans'] | oui |
| `'sur'` | ['sur'] | oui |
| `' sur'` | ['Ġsur'] | oui |
| `'par'` | ['par'] | oui |
| `' par'` | ['Ġpar'] | oui |
| `'pour'` | ['pour'] | oui |
| `' pour'` | ['Ġpour'] | oui |
| `'sans'` | ['sans'] | oui |
| `' sans'` | ['Ġsans'] | oui |
| `'qui'` | ['qui'] | oui |
| `' qui'` | ['Ġqui'] | oui |
| `'dont'` | ['dont'] | oui |
| `' dont'` | ['Ġdont'] | oui |
| `'avant'` | ['av', 'ant'] | oui |
| `' avant'` | ['Ġavant'] | oui |
| `'aussi'` | ['auss', 'i'] | oui |
| `' aussi'` | ['Ġaussi'] | oui |
| `'plus'` | ['plus'] | oui |
| `' plus'` | ['Ġplus'] | oui |
| `'tout'` | ['tout'] | oui |
| `' tout'` | ['Ġtout'] | oui |
| `'tous'` | ['t', 'ous'] | oui |
| `' tous'` | ['Ġtous'] | oui |
| `'sont'` | ['s', 'ont'] | oui |
| `' sont'` | ['Ġsont'] | oui |
| `'ont'` | ['ont'] | oui |
| `' ont'` | ['Ġont'] | oui |
| `'font'` | ['font'] | oui |
| `' font'` | ['Ġfont'] | oui |
| `'vont'` | ['v', 'ont'] | oui |
| `' vont'` | ['Ġvont'] | oui |
| `'à'` | ['Ãł'] | oui |
| `' à'` | ['ĠÃł'] | oui |
| `'où'` | ['o', 'Ã¹'] | oui |
| `' où'` | ['ĠoÃ¹'] | oui |
| `'là'` | ['lÃł'] | oui |
| `' là'` | ['ĠlÃł'] | oui |
| `'ça'` | ['Ã§a'] | oui |
| `' ça'` | ['ĠÃ§a'] | oui |

## `Qwen/Qwen3-0.6B`

- Taille du vocabulaire : 151669 ; tokens interdits : **48351** (31.9%)
- Tokens spéciaux autorisés : ['<|endoftext|>', '<|im_end|>']
- Lettres `àâçîïôùûüÿÀÂÇÎÏÔÙÛÜŸ` représentables (seules et précédées d'un espace) : **oui**

50 tokens autorisés les plus courts (mots ≥ 2 lettres) : 'in', 'on', 'at', 'st', 'or', 'it', 'an', 'ar', 'al', 'ou', 'is', 'ic', 'ro', 'as', 'ct', 'nd', 'id', 'am', 'om', 'im', 'il', 'ur', 'ad', 'ch', 'ut', 'if', 'ol', 'th', 'ig', 'iv', 'od', 'ag', 'ay', 'ot', 'us', 'un', 'ul', 'ow', 'ab', 'um', 'tr', 'ck', 'lo', 'ap', 'ub', 'ly', 'av', 'ir', 'nt', 'pt'

50 tokens autorisés les plus fréquents (approximation : ordre des fusions BPE) : 'in', 'on', 'at', 'st', 'or', ' th', 'it', 'an', 'ar', 'al', 'ou', 'is', 'ing', 'ion', 'ic', 'ro', 'as', 'ct', 'nd', ' in', 'id', 'am', ' to', ' of', 'om', 'im', 'il', ' and', 'ur', 'ad', 'ch', 'ut', 'if', 'ol', 'th', 'ig', 'iv', 'od', 'ag', 'ay', 'ot', 'us', ' st', 'un', 'ul', 'ow', 'ation', ' for', 'ab', 'ort'

| Mot outil | Tokens | Tous autorisés |
|---|---|---|
| `'un'` | ['un'] | oui |
| `' un'` | ['Ġun'] | oui |
| `'au'` | ['au'] | oui |
| `' au'` | ['Ġau'] | oui |
| `'du'` | ['du'] | oui |
| `' du'` | ['Ġdu'] | oui |
| `'ma'` | ['ma'] | oui |
| `' ma'` | ['Ġma'] | oui |
| `'mon'` | ['mon'] | oui |
| `' mon'` | ['Ġmon'] | oui |
| `'son'` | ['son'] | oui |
| `' son'` | ['Ġson'] | oui |
| `'nos'` | ['nos'] | oui |
| `' nos'` | ['Ġnos'] | oui |
| `'vos'` | ['vos'] | oui |
| `' vos'` | ['Ġvos'] | oui |
| `'ou'` | ['ou'] | oui |
| `' ou'` | ['Ġou'] | oui |
| `'mais'` | ['ma', 'is'] | oui |
| `' mais'` | ['Ġmais'] | oui |
| `'donc'` | ['don', 'c'] | oui |
| `' donc'` | ['Ġdonc'] | oui |
| `'car'` | ['car'] | oui |
| `' car'` | ['Ġcar'] | oui |
| `'ainsi'` | ['ains', 'i'] | oui |
| `' ainsi'` | ['Ġainsi'] | oui |
| `'dans'` | ['d', 'ans'] | oui |
| `' dans'` | ['Ġdans'] | oui |
| `'sur'` | ['sur'] | oui |
| `' sur'` | ['Ġsur'] | oui |
| `'par'` | ['par'] | oui |
| `' par'` | ['Ġpar'] | oui |
| `'pour'` | ['pour'] | oui |
| `' pour'` | ['Ġpour'] | oui |
| `'sans'` | ['sans'] | oui |
| `' sans'` | ['Ġsans'] | oui |
| `'qui'` | ['qui'] | oui |
| `' qui'` | ['Ġqui'] | oui |
| `'dont'` | ['dont'] | oui |
| `' dont'` | ['Ġdont'] | oui |
| `'avant'` | ['av', 'ant'] | oui |
| `' avant'` | ['Ġavant'] | oui |
| `'aussi'` | ['auss', 'i'] | oui |
| `' aussi'` | ['Ġaussi'] | oui |
| `'plus'` | ['plus'] | oui |
| `' plus'` | ['Ġplus'] | oui |
| `'tout'` | ['tout'] | oui |
| `' tout'` | ['Ġtout'] | oui |
| `'tous'` | ['t', 'ous'] | oui |
| `' tous'` | ['Ġtous'] | oui |
| `'sont'` | ['s', 'ont'] | oui |
| `' sont'` | ['Ġsont'] | oui |
| `'ont'` | ['ont'] | oui |
| `' ont'` | ['Ġont'] | oui |
| `'font'` | ['font'] | oui |
| `' font'` | ['Ġfont'] | oui |
| `'vont'` | ['v', 'ont'] | oui |
| `' vont'` | ['Ġvont'] | oui |
| `'à'` | ['Ãł'] | oui |
| `' à'` | ['ĠÃł'] | oui |
| `'où'` | ['o', 'Ã¹'] | oui |
| `' où'` | ['ĠoÃ¹'] | oui |
| `'là'` | ['lÃł'] | oui |
| `' là'` | ['ĠlÃł'] | oui |
| `'ça'` | ['Ã§a'] | oui |
| `' ça'` | ['ĠÃ§a'] | oui |

## `hf-internal-testing/llama-tokenizer`

- Taille du vocabulaire : 32000 ; tokens interdits : **12472** (39.0%)
- Tokens spéciaux autorisés : ['</s>']
- Lettres `àâçîïôùûüÿÀÂÇÎÏÔÙÛÜŸ` représentables (seules et précédées d'un espace) : **oui**

50 tokens autorisés les plus courts (mots ≥ 2 lettres) : 'in', 'on', 'at', 'or', 'an', 'is', 'it', 'ar', 'ou', 'al', 'om', 'ic', 'as', 'nd', 'st', 'ch', 'ro', 'il', 'ct', 'am', 'ol', 'im', 'ot', 'ad', 'ut', 'ur', 'id', 'ig', 'ra', 'qu', 'ow', 'un', 'ag', 'ul', 'os', 'if', 'ly', 'ab', 'ri', 'us', 'ir', 'ck', 'th', 'ay', 'od', 'um', 'ht', 'pp', 'pt', 'lo'

50 tokens autorisés les plus fréquents (approximation : ordre des fusions BPE) : 'in', 'on', ' th', 'at', 'or', 'an', 'is', 'it', 'ar', 'ou', 'al', 'om', 'ion', 'ing', 'ic', 'as', ' in', 'nd', 'st', ' to', 'ch', 'ro', 'il', ' of', 'ct', 'am', ' and', 'ol', 'im', 'ot', 'ad', 'ut', 'ur', 'id', 'ig', 'ra', ' is', 'qu', 'ow', 'un', 'ag', 'ul', 'os', 'if', 'ation', ' for', ' you', 'ly', 'ab', ' it'

| Mot outil | Tokens | Tous autorisés |
|---|---|---|
| `'un'` | ['▁un'] | oui |
| `' un'` | ['▁un'] | oui |
| `'au'` | ['▁au'] | oui |
| `' au'` | ['▁au'] | oui |
| `'du'` | ['▁du'] | oui |
| `' du'` | ['▁du'] | oui |
| `'ma'` | ['▁ma'] | oui |
| `' ma'` | ['▁ma'] | oui |
| `'mon'` | ['▁mon'] | oui |
| `' mon'` | ['▁mon'] | oui |
| `'son'` | ['▁son'] | oui |
| `' son'` | ['▁son'] | oui |
| `'nos'` | ['▁nos'] | oui |
| `' nos'` | ['▁nos'] | oui |
| `'vos'` | ['▁v', 'os'] | oui |
| `' vos'` | ['▁v', 'os'] | oui |
| `'ou'` | ['▁ou'] | oui |
| `' ou'` | ['▁ou'] | oui |
| `'mais'` | ['▁mais'] | oui |
| `' mais'` | ['▁mais'] | oui |
| `'donc'` | ['▁donc'] | oui |
| `' donc'` | ['▁donc'] | oui |
| `'car'` | ['▁car'] | oui |
| `' car'` | ['▁car'] | oui |
| `'ainsi'` | ['▁ainsi'] | oui |
| `' ainsi'` | ['▁ainsi'] | oui |
| `'dans'` | ['▁dans'] | oui |
| `' dans'` | ['▁dans'] | oui |
| `'sur'` | ['▁sur'] | oui |
| `' sur'` | ['▁sur'] | oui |
| `'par'` | ['▁par'] | oui |
| `' par'` | ['▁par'] | oui |
| `'pour'` | ['▁pour'] | oui |
| `' pour'` | ['▁pour'] | oui |
| `'sans'` | ['▁sans'] | oui |
| `' sans'` | ['▁sans'] | oui |
| `'qui'` | ['▁qui'] | oui |
| `' qui'` | ['▁qui'] | oui |
| `'dont'` | ['▁dont'] | oui |
| `' dont'` | ['▁dont'] | oui |
| `'avant'` | ['▁avant'] | oui |
| `' avant'` | ['▁avant'] | oui |
| `'aussi'` | ['▁aussi'] | oui |
| `' aussi'` | ['▁aussi'] | oui |
| `'plus'` | ['▁plus'] | oui |
| `' plus'` | ['▁plus'] | oui |
| `'tout'` | ['▁tout'] | oui |
| `' tout'` | ['▁tout'] | oui |
| `'tous'` | ['▁tous'] | oui |
| `' tous'` | ['▁tous'] | oui |
| `'sont'` | ['▁sont'] | oui |
| `' sont'` | ['▁sont'] | oui |
| `'ont'` | ['▁ont'] | oui |
| `' ont'` | ['▁ont'] | oui |
| `'font'` | ['▁font'] | oui |
| `' font'` | ['▁font'] | oui |
| `'vont'` | ['▁v', 'ont'] | oui |
| `' vont'` | ['▁v', 'ont'] | oui |
| `'à'` | ['▁à'] | oui |
| `' à'` | ['▁à'] | oui |
| `'où'` | ['▁où'] | oui |
| `' où'` | ['▁où'] | oui |
| `'là'` | ['▁là'] | oui |
| `' là'` | ['▁là'] | oui |
| `'ça'` | ['▁', 'ça'] | oui |
| `' ça'` | ['▁', 'ça'] | oui |
