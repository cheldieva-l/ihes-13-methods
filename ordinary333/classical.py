"""Exact native-permutation path tools. Smaller observed lengths, not optimum claims."""
import json
from pathlib import Path
import numpy as np


def load_moves():
    """Read exact native signed54 permutations; return name-to-pullback arrays."""
    g=json.loads((Path(__file__).parent/'assets/geometry.json').read_text())
    return {n:np.array(p) for n,p in zip(g['names'],g['moves'])}


def inverse_word(word):
    """Reverse order and invert every signed quarter turn."""
    return [x[1:] if x.startswith('-') else '-'+x for x in reversed(word)]


def replay(state,word,moves):
    """Apply a route to all54 distinct labels, including centers."""
    state=np.array(state,dtype=np.uint8)
    for token in word:state=state[moves[token]]
    return state


def reduce_axis_runs(word):
    """Commute parallel slices, reduce each signed exponent modulo4, iterate to stability."""
    old=None;word=list(word)
    while word!=old:
        old=word;word=[];i=0
        while i<len(old):
            axis=old[i].lstrip('-')[0];counts=[0,0,0]
            while i<len(old) and old[i].lstrip('-')[0]==axis:
                counts[int(old[i][-1])]+= -1 if old[i].startswith('-') else 1;i+=1
            for layer,value in enumerate(counts):
                power=value%4;base=axis+str(layer)
                word.extend([base]*power if power<3 else ['-'+base])
    return word


def exact_dictionary(moves,depth=6):
    """Full BFS ball: maps exact permutations to shortest words. Radius6 needs several GB."""
    table={bytes(range(54)):()};front=[bytes(range(54))]
    for level in range(depth):
        following=[]
        for key in front:
            state=np.frombuffer(key,dtype=np.uint8)
            for name,move in moves.items():
                new=state[move].tobytes()
                if new not in table:table[new]=table[key]+(name,);following.append(new)
        front=following
    return table


def shorten(word,moves,table):
    """Replace the most profitable exact-equivalent subword until no strict gain exists."""
    word=reduce_axis_runs(word)
    while True:
        chosen=None
        for start in range(len(word)):
            state=np.arange(54,dtype=np.uint8)
            for stop in range(start,len(word)):
                state=state[moves[word[stop]]];replacement=table.get(state.tobytes())
                if replacement is not None and len(replacement)<stop-start+1:
                    gain=stop-start+1-len(replacement)
                    if chosen is None or gain>chosen[0]:chosen=(gain,start,stop+1,replacement)
        if chosen is None:return word
        _,start,stop,replacement=chosen
        word=reduce_axis_runs(word[:start]+list(replacement)+word[stop:])
