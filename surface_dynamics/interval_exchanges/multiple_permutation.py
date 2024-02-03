r"""
Multiple permutations

A multiple permutations is the combinatoric part of a multiple circle (or interval) exchange
transformation, which is an exchange transformation (IET) on several circles.
Those maps can be studied just as interval exchange transformation, but it
can be convenient to define them on several components, in particular for
the study of tiling billiards, see [BSDFI18]_.

AUTHORS:

- Magali Jay (2024-02-02) : initial version

TESTS::

    sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
    sage: mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]])
    [[['a', 'b', 'c'], ['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]]
    sage: mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]],flips=['a','c','e'])
    [[['a', 'b', 'c'], ['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]]
"""
#*****************************************************************************
#       Copyright (C) 2024 Magali Jay <magali.jay@ens-paris-saclay.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
#*****************************************************************************


# TODO : ajouter tests avec perm dont une des comp n'a pas même nb de labels en haut et en bas

class MultiplePermutation:
    def __init__(self, intervals=None, reduced=None, flips=None, alphabet=None):
        r"""
        Returns a permutation of a multiple circle (or interval) exchange transformation.
    
        Those permutations are the combinatoric part of a multiple circle (or interval) exchange
        transformation, which is an exchange transformation (IET) on several circles. 
        The combinatorial study of those objects starts with Gerard Rauzy [Rau80]_ 
        and William Veech [Vee78]_.
    
        The combinatoric part of interval exchange transformation can be taken
        independently from its dynamical origin. It has an important link with
        strata of Abelian differential (see ?)
    
        INPUT:
    
        - ``intervals`` - a list of length k (for a multiple permutation with k components)
            whose entries are lists of length 2, representing the top and the bottom interval.
            To be implemented: support of string, two strings, list or tuples that can be
            converted to two lists.
    
        - ``reduced`` - boolean (default: False) specifies reduction. False means
          labelled permutation and True means reduced permutation.
    
        - ``flips`` -  iterable (default: None) the letters which correspond to
          flipped intervals.
    
        - ``alphabet`` - (optional) something that should be converted to an alphabet.
    
    
        EXAMPLES::
    
            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
    
        Creation of multiple permutations. Note: The `__repr__` method is not user friendly yet. ::
    
            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]])
            [[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]]
        
            sage: mp.MultiplePermutation([[[0,1,2,3],[2,1,3,0]],[[5],[5]]])
            [[[0, 1, 2, 3], [2, 1, 3, 0]], [[5], [5]]]
            
            sage: l1 = [[0, 'A', 'B', 1], ['B', 0, 1, 'A']]
            sage: l2 = [['a1', 'b1', 'c1'],['c1', 'a2', 'b1']]
            sage: l3 = [['a2','b2','c2'],['c2','a1','b2']]
            sage: mp.MultiplePermutation([l1, l2, l3])
            [[[0, 'A', 'B', 1], ['B', 0, 1, 'A']], [['a1', 'b1', 'c1'], ['c1', 'a2', 'b1']], [['a2', 'b2', 'c2'], ['c2', 'a1', 'b2']]]

            sage: mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]])
            [[['a', 'b', 'c'], ['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]]
    
        Creation of reduced permutations. Not implemented yet ::
    
            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]], reduced=True) #[[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]]
            Traceback (most recent call last):
            ...
            ValueError: Not implemented yet
        
            sage: mp.MultiplePermutation([[[0,1,2,3],[2,1,3,0]],[[5],[5]]], reduced=True)
            Traceback (most recent call last):
            ...
            ValueError: Not implemented yet

            sage: l1 = [[0, 'A', 'B', 1], ['B', 0, 1, 'A']]
            sage: l2 = [['a1', 'b1', 'c1'],['c1', 'a2', 'b1']]
            sage: l3 = [['a2','b2','c2'],['c2','a1','b2']]
            sage: mp.MultiplePermutation([l1, l2, l3], reduced=True)
            Traceback (most recent call last):
            ...
            ValueError: Not implemented yet

    
        Managing the alphabet: two labelled permutations with different (ordered)
        alphabet but with the same labels are different. ::
    
            sage: p = iet.Permutation('a b','b a', alphabet='ab')
            sage: q = iet.Permutation('a b','b a', alphabet='ba')
            sage: str(p) == str(q)
            True
            sage: p == q
            False
    
        For reduced permutations, the alphabet does not play any role excepted for
        printing the object. Not implemented yet (it works, but why?) ::
    
            sage: p = iet.Permutation([['a', 'b',  'c'], ['c', 'b',  'a']], reduced=True)
            sage: q = iet.Permutation([[0,1,2],[2,1,0]], reduced=True)
            sage: p == q
            True
    
        Creation of flipped permutations. The `_repr_` method does not take into account the flips::
    
            sage: mp.MultiplePermutation([[['a', 'b',  'c'], ['c', 'b',  'a']]], flips=['a','b'])
            [[['a', 'b',  'c'], ['c', 'b',  'a']]]
            sage: mp.MultiplePermutation([[['a', 'b',  'c'], ['c', 'b',  'a']]], flips='ab', reduced=True)
            Traceback (most recent call last):
            ...
            ValueError: Not implemented yet
    
        TESTS::
    
            sage: type(mp.MultiplePermutation([[['a', 'b',  'c'], ['c', 'b',  'a']]], reduced=True))
            Traceback (most recent call last):
            ...
            ValueError: Not implemented yet
            sage: type(mp.MultiplePermutation([[['a', 'b',  'c'], ['c', 'b',  'a']]], reduced=False))
            <class 'surface_dynamics.interval_exchanges.multiple_permutation.MultiplePermutation'>
            sage: type(mp.MultiplePermutation([[['a', 'b',  'c'], ['c', 'b',  'a']]], reduced=True, flips=['a','b']))
            Traceback (most recent call last):
            ...
            ValueError: Not implemented yet
            sage: type(mp.MultiplePermutation([[['a', 'b',  'c'], ['c', 'b',  'a']]], reduced=False, flips=['a','b']))
            <class 'surface_dynamics.interval_exchanges.multiple_permutation.MultiplePermutation'>
    
            sage: p = iet.Permutation(('a b c','c b a'))
            sage: iet.Permutation(p) == p
            True
            sage: q = iet.Permutation(p, reduced=True)
            sage: q == p
            False
            sage: q == p.reduced()
            True
    
            sage: p = iet.Permutation('a', 'a', flips='a', reduced=True)
            sage: iet.Permutation(p) == p
            True
    
            sage: p = iet.Permutation('a b c','c b a',flips='a')
            sage: iet.Permutation(p) == p
            True
            sage: iet.Permutation(p, reduced=True) == p.reduced()
            True
    
            sage: p = iet.Permutation('a b c','c b a',reduced=True)
            sage: iet.Permutation(p) == p
            True
        """
        # this constructor assumes that several methods are present
        #  _init_twin(intervals)
        #  _set_alphabet(alphabet)
        #  _init_alphabet(intervals)
        #  _init_flips(intervals, flips)
        from .constructors import _two_lists
        if intervals is not None :
            for k in range(len(intervals)):
                if not isinstance(intervals[k],list) or len(intervals[k]) == 1 :
                    intervals[k] = _two_lists(intervals[k],None)
                elif len(intervals[k]) == 2 :
                    intervals[k] = _two_lists(intervals[k][0],intervals[k][1])
                else :
                    raise ValueError('The data of intervals has not the expected shape')
        
        # setting twins
        if intervals is None:
            self._twin_component = [[], []]
            self._twin_position = [[], []]
        else:
            self._init_twin(intervals)

        # setting alphabet
        if alphabet is not None:
            self._set_alphabet(alphabet)
        elif intervals is not None:
            self._init_alphabet(intervals)

        # optionally setting labels
        if intervals is not None and not reduced:
            ncomp = len(intervals)
            
            self._labels = [[
                list(map(self._alphabet.rank, intervals[k][0])),
                list(map(self._alphabet.rank, intervals[k][1]))]
                           for k in range(ncomp)]

        # optionally setting flips # question : optional or not ?
        if flips is not None:
            self._init_flips(intervals, flips)
        else:
            self._flips = None

        if reduced:
            raise ValueError('Not implemented yet')



    def _init_twin(self, a):
        r"""
        Initializes the twin list.

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

            sage: mp.MultiplePermutation([[[0,1,2,3],[2,1,3,0]],[[5],[5]]]) #indirect doctest
            [[[0, 1, 2, 3], [2, 1, 3, 0]], [[5], [5]]]
        """
        if a is None:
            self._twin_component = [[],[]]
            self._twin_position = [[],[]]

        else:
            ncomp = len(a)
            self._twin_component = [[[0]*len(a[i][0]),[0]*len(a[i][1])] for i in range(ncomp)]
            self._twin_position = [[[0]*len(a[i][0]),[0]*len(a[i][1])] for i in range(ncomp)]
            for k in range(ncomp) :
                for i in range(len(a[k][0])) :
                    k_twin = 0
                    while k_twin < ncomp and a[k][0][i] not in a[k_twin][1]:
                        k_twin += 1

                    if k_twin == ncomp :
                        raise ValueError('Letter %s should appear in bottom intervals' % a[k][0][i])
                    else :
                        for k_other in range(k_twin+1, ncomp):
                            if a[k][0][i] in a[k_other][1]:
                                raise ValueError('Letter %s should appear exactly once in bottom intervals' % a[k][0][i])

                    j = a[k_twin][1].index(a[k][0][i])
                    self._twin_component[k][0][i] = k_twin
                    self._twin_position[k][0][i] = j
                    self._twin_component[k_twin][1][j] = k
                    self._twin_position[k_twin][1][j] = i


    def _set_alphabet(self, alphabet): # question : copié-collé, comment faire ref à l'original ?
        r"""
        Sets the alphabet of self.

        TESTS:

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

            sage: p = mp.MultiplePermutation([[['a', 'b'], ['a','b']]])
            sage: p.alphabet([0,1])   #indirect doctest
            sage: p.alphabet() == Alphabet([0,1])
            True
            sage: p
            [[[0, 1], [0, 1]]]
            sage: p.alphabet("cd")   #indirect doctest
            sage: p.alphabet() == Alphabet(['c','d'])
            True
            sage: p
            [[['c', 'd'], ['c', 'd']]]

            sage: p = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]])
            sage: p.alphabet([0,1])
            Traceback (most recent call last):
            ...
            ValueError: not enough letters in alphabet

            sage: p = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]])
            sage: p.alphabet([1,2,3,4,5]) #indirect doctest
            sage: p
            [[[1, 2, 3], [3, 5, 2, 4]], [[4, 5], [1]]]
    
        Tests with reduced permutations. Not implemented yet::

            sage: p = mp.MultiplePermutation([[['a', 'b'], ['a','b']]],reduced=True).alphabet()
            Traceback (most recent call last):
            ...
            ValueError: Not implemented yet
        """
        from sage.combinat.words.alphabet import build_alphabet
        alphabet = build_alphabet(alphabet)
        if alphabet.cardinality() < len(self):
            raise ValueError("not enough letters in alphabet")
        self._alphabet = alphabet

    
    def _init_alphabet(self,a) :
        r"""
        Initializes the alphabet from intervals.

        INPUT:

        - ``a`` - the two intervals as lists

        TESTS::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            
            sage: p1 = mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]])  #indirect doctest
            sage: p1.alphabet() == Alphabet(['a', 'b', 'c', 'd'])
            True
        
            sage: p2 = mp.MultiplePermutation([[[0,1,2,3],[2,1,3,0]],[[4],[4]]])  #indirect doctest
            sage: p2.alphabet() == Alphabet([0, 1, 2, 3, 4])
            True
        """
        from sage.combinat.words.alphabet import build_alphabet
        letters = []
        for top, bot in a:
            letters += top
        self._alphabet = build_alphabet(letters)

    def alphabet(self, data=None):
        r"""
        Manages the alphabet of the multiple permutation.

        If there is no argument, the method returns the alphabet used. If the
        argument could be converted to an alphabet, this alphabet will be used.

        INPUT:

        - ``data`` - None or something that could be converted to an alphabet


        OUTPUT:

        - either None or the current alphabet


        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

            sage: p = mp.MultiplePermutation([[['a', 'b'], ['a', 'b']]])
            sage: p
            [[['a', 'b'], ['a', 'b']]]
            sage: p.alphabet([0,1])
            sage: p.alphabet() == Alphabet([0,1])
            True
            sage: p
            [[[0, 1], [0, 1]]]
            sage: p.alphabet("cd")
            sage: p.alphabet() == Alphabet(['c','d'])
            True
            sage: p
            [[['c', 'd'], ['c', 'd']]]
        """
        if data is None:
            return self._alphabet
        else:
            self._set_alphabet(data)
    
    def _init_flips(self,intervals,flips):
        r"""
        Initialize the flip list

        TESTS::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]]).flips() #indirect doctest
            
            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]], flips='b').flips() #indirect doctest
            ['b']
            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]], flips='ab').flips() #indirect doctest
            ['a', 'b']
            sage: mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed').flips() #indirect doctest
            ['d', 'e']

        """
        ncomp = len(intervals)
        self._flips = [[[1]*len(intervals[k][0]), [1]*len(intervals[k][1])] for k in range(ncomp)]
        if flips is not None : # should be the case (if flips is None, we do not initialize ._flips) 
            for k in range(ncomp):
                for level in (0,1):
                    for i,letter in enumerate(intervals[k][level]):
                        if letter in flips:
                            self._flips[k][level][i] = -1


    def flips(self):
        r"""
        Returns the list of flips.

        If the permutation is not a flipped permutation then ``None`` is returned.

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]]).flips() is None
            True
            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]],flips=[]).flips()
            []
            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]], flips='b').flips()
            ['b']
            sage: mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]], flips='ab').flips()
            ['a', 'b']
            sage: mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed').flips()
            ['d', 'e']
        """
        if self._flips is not None:
            letters = []
            for k in range(self.number_of_components()):
                for i,f in enumerate(self._flips[k][0]):
                    if f == -1 : #and self._labels[k][0][i]] not in letters:
                        # utiliser map?
                        lab = self._labels[k][0][i]
                        flipped_letter = self._alphabet.unrank(lab)
                        letters.append(flipped_letter)
            return letters

    
    def list(self, flips=False):
        r"""
        Returns a list whose entries are lists of two lists corresponding to the intervals.

        INPUT:

        - ``flips`` - boolean (default: False) - if ``True`` returns instead of
          letters use pair of letter and flip.

        OUTPUT: a list of k lists of two lists of labels (or labels with flips),
        where k is the number of components.

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
    
        The list of a multiple permutation::

            sage: p1 = mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]])
            sage: p1.list()
            [[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]]
        
            sage: p2 = mp.MultiplePermutation([[[0,1,2,3],[2,1,3,0]],[[5],[5]]])
            sage: p2.list()
            [[[0, 1, 2, 3], [2, 1, 3, 0]], [[5], [5]]]

            sage: p1 = mp.MultiplePermutation([[[1, 2, 3], [5, 1, 2]], [[4, 5], [3, 4]]])
            sage: p1.list()
            [[[1, 2, 3], [5, 1, 2]], [[4, 5], [3, 4]]]
            sage: p1.alphabet("abcde")
            sage: p1.list()
            [[['a', 'b', 'c'], ['e', 'a', 'b']], [['d', 'e'], ['c', 'd']]]

            sage: p2 = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]])
            sage: p2.list()
            [[['a', 'b', 'c'], ['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]]

        Recovering the permutation from this list (and the alphabet)::

            sage: q1 = mp.MultiplePermutation(p1.list(),alphabet=p1.alphabet())
            sage: p1 == q1
            True

            sage: q2 = mp.MultiplePermutation(p2.list(),alphabet=p2.alphabet())
            sage: p2 == q2
            True
            
        Some examples with flips::

            sage: p = mp.MultiplePermutation([[[1, 2, 3], [5, 1, 2]], [[4, 5], [3, 4]]], flips=[1])
            sage: p.list(flips=True)
            [[[(1, -1), (2, 1), (3, 1)], [(5, 1), (1, -1), (2, 1)]],
             [[(4, 1), (5, 1)], [(3, 1), (4, 1)]]]
            sage: p.list(flips=False)
            [[[1, 2, 3], [5, 1, 2]], [[4, 5], [3, 4]]]

            sage: p2 = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed')
            sage: p2.list(flips=True)
            [[[('a', 1), ('b', 1), ('c', 1)], [('c', 1), ('e', -1), ('b', 1), ('d', -1)]],
             [[('d', -1), ('e', -1)], [('a', 1)]]]          
            sage: p2 = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed')
            sage: p2.list(flips=False)
            [[['a', 'b', 'c'], ['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]]

            sage: mp.MultiplePermutation([[[1, 2, 3], [5, 1, 2]], [[4, 5], [3, 4]]]).list(flips=True)
            [[[(1, 1), (2, 1), (3, 1)], [(5, 1), (1, 1), (2, 1)]],
             [[(4, 1), (5, 1)], [(3, 1), (4, 1)]]]


        The list can be used to reconstruct the permutation::

            sage: p = mp.MultiplePermutation([[['a', 'b', 'c'], ['c', 'a', 'b']]])
            sage: p == mp.MultiplePermutation(p.list())
            True
            
            sage: p = mp.MultiplePermutation([[['a', 'b', 'c'], ['c', 'a', 'b']]],flips='ab')
            sage: p == mp.MultiplePermutation(p.list(), flips=p.flips())
            True

            sage: p2 = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed')
            sage: p2 == mp.MultiplePermutation(p2.list(), flips=p2.flips())
            True
        """
        if flips:
            if self._flips is None:
                flips = [[[1] * len(self._labels[k][0]), [1] * len(self._labels[k][1])] for k in range(self.number_of_components())]
            else:
                flips = self._flips
            a = []
            for k in range(self.number_of_components()):
                a0 = list(zip(map(self._alphabet.unrank, self._labels[k][0]), flips[k][0]))
                a1 = list(zip(map(self._alphabet.unrank, self._labels[k][1]), flips[k][1]))
                a.append([a0,a1])
        else:
            a = []
            for k in range(self.number_of_components()):
                a0 = list(map(self._alphabet.unrank, self._labels[k][0]))
                a1 = list(map(self._alphabet.unrank, self._labels[k][1]))
                a.append([a0,a1])

        return a


    def number_of_components(self) :
        r"""
        Returns the number of component of the multiple permutation.

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

            sage: p = mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]])
            sage: p.number_of_components()
            1
            
            sage: p = mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]], flips='b')
            sage: p.number_of_components()
            1
            
            sage: l1 = [[0, 'A', 'B', 1], ['B', 0, 1, 'A']]
            sage: l2 = [['a1', 'b1', 'c1'],['c1', 'a2', 'b1']]
            sage: l3 = [['a2','b2','c2'],['c2','a1','b2']]
            sage: p = mp.MultiplePermutation([l1, l2, l3])
            sage: p.number_of_components()
            3

            sage: p2 = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed')
            sage: p2.number_of_components()
            2
        """
        return len(self._twin_component)

    
    # TODO ? list whose coordinates are the lengths of each permutation component of the multiple permutation. TOP and BOT
    def __len__(self) :
        r"""
        Returns the total number of labels in the multiple permutation.

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

            sage: p = mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]])
            sage: len(p)
            4
            
            sage: p = mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]], flips='b')
            sage: len(p)
            4
            
            sage: l1 = [[0, 'A', 'B', 1], ['B', 0, 1, 'A']]
            sage: l2 = [['a1', 'b1', 'c1'],['c1', 'a2', 'b1']]
            sage: l3 = [['a2','b2','c2'],['c2','a1','b2']]
            sage: p = mp.MultiplePermutation([l1, l2, l3])
            sage: len(p)
            10

            sage: p2 = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed')
            sage: len(p2)
            5
        """
        return sum([len(self._twin_component[k][0]) for k in range(self.number_of_components())])

    
    def __copy__(self) :
        r"""
        Returns a copy of this multiple permutation.

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

        Note: equality is not emplemented yet.

            sage: p = mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]])
            sage: q = copy(p)
            sage: p == q
            True
            sage: p is q
            False

            sage: p = mp.MultiplePermutation([[['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']]],flips=['ac'])
            sage: q = copy(p)
            sage: p == q
            True
            sage: p is q
            False
            
            sage: l1 = [[0, 'A', 'B', 1], ['B', 0, 1, 'A']]
            sage: l2 = [['a1', 'b1', 'c1'],['c1', 'a2', 'b1']]
            sage: l3 = [['a2','b2','c2'],['c2','a1','b2']]
            sage: p = mp.MultiplePermutation([l1, l2, l3])
            sage: q = copy(p)
            sage: p == q
            True
            sage: p is q
            False

            sage: p2 = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed')
            sage: q2 = copy(p2)
            sage: q2 == p2
            True
            sage: q2 is p2
            False
        """
        return self.__class__(intervals = self.list(), flips = self.flips(), alphabet = self._alphabet)
        #for k in range(self.number_of_components()) : #(len(self._labels)) :
            #print(type(self._labels[k][0]))
            # copy ne marche pas
         #   res._labels.append([list(self._labels[k][0]),list(self._labels[k][1])])
          #  res._twin_component.append([list(self._twin_component[k][0]),list(self._twin_component[k][1])])
           # res._twin_position.append([list(self._twin_position[k][0]),list(self._twin_position[k][1])])
            #res._flips.append([list(self._flips[k][0]),list(self._flips[k][1])])
        #return res


    def __eq__(self, other):
        return type(self) == type(other) and \
               self._twin_component == other._twin_component and \
               self._twin_position == other._twin_position and \
               self._labels == other._labels and \
               self._flips == other._flips and \
               (self._labels is None or self._alphabet == other._alphabet)
        

    def __ne__(self, other):
        return not self == other

    
    def __repr__(self):
        r"""
        Representation method of self.

        Apply the function str to _repr_type(_repr_options) if _repr_type is
        callable and _repr_type else.

        TESTS::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp

            sage: p = mp.MultiplePermutation([[['a'],['a']],[['b', 'c'],['c', 'd']],[['d','e'],['b','e']]])
            sage: p._repr_type = 'str'
            sage: p._repr_options = ('\n',)
            sage: p   #indirect doctest
            [[['a'], ['a']], [['b', 'c'], ['c', 'd']], [['d', 'e'], ['b', 'e']]]

        """
        # TODO : use .flips() / ajouter traitement avec flips=True pour afficher des - 
        s = []
        alph = self._alphabet
        for k in range(len(self._labels)):
            s_k0 = [alph.unrank(self._labels[k][0][i]) for i in range(len(self._labels[k][0]))]
            s_k1 = [alph.unrank(self._labels[k][1][i]) for i in range(len(self._labels[k][1]))]
            s.append([s_k0,s_k1])
        return repr(s)
    
    #return mp._labels
#            a    b c    d e
#            a    c d    c e
    #
#        if self._repr_type is None:
#            return ''
#
#        elif self._repr_type == 'reduced':
#            return ''.join(map(str,self[1]))"
#
#        else:
#            f = getattr(self, self._repr_type)
#            if callable(f):
#                return str(f(*self._repr_options))
#            else:
#                return str(f)
#
#            sage: mp._repr_options = (' / ',)
#            sage: mp   #indirect doctest
#            a b c / c b a
#        ::
#
#            sage: p._repr_type = '_twin'
#            sage: p   #indirect doctest
#            [[2, 1, 0], [2, 1, 0]]
    
    _repr_type = 'str'
    _repr_options = ("\n",)


    def shape(self):
        # TODO doc test
        s = []
        for k in range(self.number_of_components()):
            s.append([len(self._labels[k][0]),len(self._labels[k][1])])
        return s

    
# top_sg_index, bot_sg_index,
    def cut_rectangle(self, interval_index, cut_right = True, new_label = None, inplace = True):
        # assume that we work with intervals, not circles
        # assume that the lengths are the same for top and bottom cutted intervals
        ncomp = self.number_of_components()
        
        if cut_right :
            a = -1
        else :
            a = 0
            
        top_label = self._labels[interval_index][0].pop(a)
        bot_label = self._labels[interval_index][1].pop(a)
        self._labels.append([[top_label],[bot_label]])
        
        top_flip = self._flips[interval_index][0].pop(a)
        bot_flip = self._flips[interval_index][1].pop(a)
        self._flips.append([[top_flip],[bot_flip]])

        
        twin_top_int = self._twin_component[interval_index][0].pop(a)
        twin_bot_int = self._twin_component[interval_index][1].pop(a)
        
        twin_top_pos = self._twin_position[interval_index][0].pop(a)
        twin_bot_pos = self._twin_position[interval_index][1].pop(a)

        if (not cut_right) and twin_top_int == interval_index :
            twin_top_pos -= 1
        if (not cut_right) and twin_bot_int == interval_index :
            twin_bot_pos -= 1
            
        if not cut_right :
            for k in range(ncomp) :
                for level in (0,1) :
                    for i in range(len(self._twin_position[k][level])) :
                        if self._twin_component[k][level][i] == interval_index :
                            self._twin_position[k][level][i] -= 1
        
        self._twin_component.append([[twin_top_int],[twin_bot_int]])
        self._twin_position.append([[twin_top_pos],[twin_bot_pos]])
        self._twin_component[twin_top_int][1][twin_top_pos] = ncomp
        self._twin_component[twin_bot_int][0][twin_bot_pos] = ncomp
        self._twin_position[twin_top_int][1][twin_top_pos] = 0
        self._twin_position[twin_bot_int][0][twin_bot_pos] = 0
        
                            

    def stack_rectangle(self, glued_label, inplace = True, towers=[]):
        # assume that we work with intervals, not circles
        # assume that the rectangle is not a cylinder, i.e. has same label on top and bottom
        ncomp = self.number_of_components()
        comp = -1 # component which we have to glue
        k = 0
        while comp == -1 and k < ncomp :
            if len(self._labels[k][1]) == 1 and self._labels[k][1][0] == glued_label :
                comp = k
            k += 1
        if comp == -1:
            raise ValueError('There is no simple rectangle with label %s at bottom' % glued_label)

        for k in range(ncomp) :
            for level in (0,1):
                for i in range(len(self._twin_component[k][level])) :
                    if self._twin_component[k][level][i] > comp :
                        self._twin_component[k][level][i] -= 1                
        
        [[image_comp],[twin_comp]] = self._twin_component.pop(comp) # twin_comp is the component on which we glue
        [[image_pos],[twin_pos]] = self._twin_position.pop(comp) # twin_pos is the position in which we glue
        [[image_label],[g_l]] = self._labels.pop(comp) # g_l =glued_labed
        [[image_flip],[glued_flip]] = self._flips.pop(comp)

        self._labels[twin_comp][0][twin_pos] = image_label
        self._twin_component[twin_comp][0][twin_pos] = image_comp
        self._twin_position[twin_comp][0][twin_pos] = image_pos
        self._flips[twin_comp][0][twin_pos] *= image_flip

        self._twin_component[image_comp][1][image_pos] = twin_comp
        self._twin_position[image_comp][1][image_pos] = twin_pos
        self._flips[image_comp][1][image_pos] *= glued_flip 

        
        p = 0
        while p < len(towers) and towers[p][-1] != glued_label :
            p += 1
        if p == len(towers) :
            towers.append([glued_label,image_label])
        else :
            towers[p].append(image_label)

        return towers

