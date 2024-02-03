# TODO : garder seulement fonctions et tests utiles, enlever import *

r"""
Multiple Exchange Transformations

A multiple exchange transformation is a map defined on several circles (or intervals). See
help(iet.MultipleExchangeTransformation) for a more complete help.

AUTHOR:

- Magali Jay (2024-02-02)


EXAMPLES::

    sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
    sage: import surface_dynamics.interval_exchanges.met as met
    
Initialization of a simple iet with integer lengths. The `_repr_` method is not implemented yet::

    sage: p = mp.MultiplePermutation([[[3,2,1], [3,1,2]]])
    sage: T = met.MultipleExchangeTransformation(p, [3,2,1], [[0,0]]) #indirect doctest
    
Initialization of a multiple iet with or without flips::

    sage: p = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]])
    sage: T = met.MultipleExchangeTransformation(p, [3,5,7,1,2], [[1,2],[1/2,3/2]]) #indirect doctest

    sage: p = mp.MultiplePermutation([[['a','b','c'],['c', 'e', 'b', 'd']], [['d', 'e'], ['a']]], flips='ed')
    sage: T = met.MultipleExchangeTransformation(p, [3,5,7,1,2], [[1,2],[1/2,3/2]]) #indirect doctest

"""
#*****************************************************************************
#       Copyright (C) 2024 Magali Jay <magali.jay@ens-paris-saclay.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
#*****************************************************************************

# question : qu'est ce qui est nécessaire ?
from __future__ import print_function, absolute_import
from six.moves import range, map, filter, zip

from copy import copy
from sage.modules.free_module_element import free_module_element
from sage.rings.all import ZZ, QQ

from .template import side_conversion, interval_conversion
from .labelled import LabelledPermutationIET

from surface_dynamics import iet
from surface_dynamics.interval_exchanges import multiple_permutation,met

class AbstractExchangeTransformation:
    def my_function(self):
        print(self._permutation)

import surface_dynamics.interval_exchanges.multiple_permutation

class MultipleExchangeTransformation: #(AbstractExchangeTransformation)
    r"""
    Multiple exchange transformation

    INPUT:

    - ``permutation`` - a list of permutations (that could be converted into a MultiplePermutation)

    - ``lengths`` - the list of each circle's or interval's list of lengths

    - ``rotation`` - the list of the list of the top and bottom base points (for interval [0, 0])

    EXAMPLES::

        sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
        sage: import surface_dynamics.interval_exchanges.met as met
    
    Direct initialization not implemented yet. Initialization from a MultiplePermutation::

        sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['c', 'b', 'a']]])
        sage: l = vector([0.5, 1, 1.2])
        sage: t = met.MultipleExchangeTransformation(perm, l, [[1/10, 4/10]])
        sage: t.permutation() == perm
        True
        sage: t.lengths() == l
        True
        sage: t.rotation()
        [3/10]

    Note that the rotation must be a list of simple rotation::

        sage: t = met.MultipleExchangeTransformation(perm, l, [1/10, 4/10])
        Traceback (most recent call last):
        ...
        ValueError: rotation must be a list of list
        
        
    If it is not possible to convert lengths to real values an error is raised. Error not implemented yet.
    The value for the lengths must be positive. Error not implemented yet.
    """
    from surface_dynamics import iet
    #from surface_dynamics.interval_exchanges import multiple_permutation
    import surface_dynamics.interval_exchanges.multiple_permutation
    
    def __init__(self, multi_perm=None, lengths=None, multi_rotation=None, base_ring=None):
        r"""
        INPUT:

        - ``permutation`` - a multiple permutation (MultiplePermutation)

        - ``lengths`` - the list of lengths

        - ``rotation`` - the list of top and bottom base points

        TESTS::

            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met
    
            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['c', 'b', 'e']], [['d', 'e'], ['a', 'd']]])
            sage: l = vector([0.5, 1, 1.2, 4, 0.5])
            sage: t = met.MultipleExchangeTransformation(perm, l, [[1/10, 4/10], [1/4, 0]])
            sage: t.permutation() == perm
            True
            sage: t.lengths() == l
            True

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['c', 'b', 'e']], [['d', 'e'], ['a', 'd']]], flips=['abcde'])
            sage: l = vector([0.5, 1, 1.2, 4, 0.5])
            sage: t = met.MultipleExchangeTransformation(perm, l, [[1/10, 4/10], [1/4, 0]])
            sage: t.permutation() == perm
            True
            sage: t.lengths() == l
            True

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['c', 'b', 'e']], [['d', 'e'], ['a', 'd']]], flips=['abcde'])
            sage: l = vector([ 1, 2, 3, 4, 1])
            sage: t = met.MultipleExchangeTransformation(perm, l, [[0, 2], [3, 0]])
            sage: a = perm.alphabet()
            sage: t.permutation() ==  mp.MultiplePermutation([[['a', 'b', 'c'], ['e', 'c', 'b']], [['e', 'd'], ['a', 'd']]], flips=['abcde'], alphabet=a)
            True
            sage: t.lengths() == l
            True
        """
        #    sage: p == loads(dumps(p))   # question
        #    True
        
        from sage.modules.free_module_element import free_module_element as vector
        
        if multi_perm is None or lengths is None or multi_rotation is None:
            self._permutation = None
            self._lengths = []
            self._rotation = []
            self._base_ring = None
            self._vector_space = None
        
        else:
            #if isinstance(lengths, dict): # question
             #   l = [0] * len(permutations)
              #  rank = permutations._alphabet.rank
               # for letter, length in lengths.items():
                #    l[rank(letter)] = length
                #lengths = l
            if not isinstance(multi_perm, surface_dynamics.interval_exchanges.multiple_permutation.MultiplePermutation):
                raise ValueError('permutation must be a MultiplePermutation')
            
            ncomp = multi_perm.number_of_components() # number of components
            nlab = len(multi_perm) # number of labels

            if len(multi_rotation) > 0 and not isinstance(multi_rotation[0],list):
                raise ValueError('rotation must be a list of list')
                
            if len(multi_rotation) != ncomp:
                raise ValueError('permutation and rotation do not match')
                
            self._lengths = vector(lengths) #[copy(vector(lengths[k])) for k in range(len(lengths))]
            #if self._lengths is lengths: 
             #   # NOTE: vector(.) never performs copy
              #  self._lengths = self._lengths.__copy__()
            l = [0] * (nlab + 2*ncomp)
            l[:nlab] = self._lengths.__copy__()
            l[nlab:nlab+ncomp] = [multi_rotation[k][0] for k in range(ncomp)]
            l[nlab+ncomp:] = [multi_rotation[k][1] for k in range(ncomp)]
            # note : we must take into account the lengths and the base points
            l = vector(l)
            self._base_ring = l.base_ring() # question : correspond à ce qu'on veut ?
            M = l.base_ring()**nlab
            self._vector_space = M.ambient_module() # question : correspond à ce qu'on veut ?
            comp_lengths = self._check_lengths(multi_perm)
            self._init_perm_and_rot(multi_perm,multi_rotation,comp_lengths)

    
    def _check_lengths(self, multi_perm) :
        shape = multi_perm.shape()
        ncomp = multi_perm.number_of_components()
        #top_cut = [sum([shape[i][0] for i in range(k)]) for k in range(ncomp+1)] # successive indices of "changement of component" for top intervals
        #bot_cut = [sum([shape[i][1] for i in range(k)]) for k in range(ncomp+1)] # successive indices of "changement of component" for bot intervals
        res = []
        for k in range(ncomp):
            top_lab_k = multi_perm._labels[k][0]
            bot_lab_k = multi_perm._labels[k][1]
            top_len_comp_k = sum([self._lengths[i] for i in top_lab_k])
            bot_len_comp_k = sum([self._lengths[i] for i in bot_lab_k])
            if (bot_len_comp_k != top_len_comp_k) :
                raise ValueError('Lengths of top and bottom intervals in component %s do not match' % k)
            res.append(top_len_comp_k)
        return res
        #top_comp_length = [sum(self._lengths[top_cut[i]:top_cut[i+1]]) for i in range(ncomp)]
        #bot_comp_length = [sum(self._lengths[bot_cut[i]:bot_cut[i+1]]) for i in range(ncomp)]
        #print(top_cut,bot_cut,top_comp_length,bot_comp_length)
        #k = 0
        #while k < ncomp :
        #    if (bot_comp_length[k] != top_comp_length[k]) :
                #raise ValueError('Lengths of top and bottom intervals in component %s do not match' % k)
            #k += 1

    # 
    def _init_perm_and_rot(self,perm,rotation,comp_lengths) :
        # initialize the permutation so that the first top (resp. bottom) label corresponds to the first complete interval after 0 
        #self._permutation = copy(perm)
        #flips = self._permutation.flips()
        ncomp = perm.number_of_components()
        #components_length = [sum(self._lengths[i]) for i in range(len(lengths))]
        new_perm = [[copy(perm._labels[k][0]), copy(perm._labels[k][1])] for k in range(ncomp)]
        
        for k in range(ncomp):
            for level in range(2):
                while rotation[k][level] < self._base_ring.zero() :
                    rotation[k][level] += comp_lengths[k]
                while rotation[k][level] >= comp_lengths[k] :
                    rotation[k][level] -= comp_lengths[k]
                ind = -1
                i = perm._labels[k][level][ind]
                x = self._lengths[i]
                c = 0
                while x <= rotation[k][level] :
                    c += 1                
                    ind -= 1
                    i = perm._labels[k][level][ind]
                    x += self._lengths[i]
                if c != 0 :
                    x -= self._lengths[i]
                    rotation[k][level] -= x
                    n = len(perm._labels[k][level])
                    labels_level = []
                    for i in range(n-c,n) :
                        labels_level.append(perm._labels[k][level][i])
                    for i in range(n-c) :
                        labels_level.append(perm._labels[k][level][i])
                    new_perm[k][level] = labels_level
                    
            for level in range(2):
                for i in range(len(new_perm[k][level])):
                    new_perm[k][level][i] = perm.alphabet().unrank(new_perm[k][level][i])
                    #new_perm[k][level][i] = self._permutation.alphabet().unrank(new_perm[level][i])
                    # new_perm contained labels that are number, after the loop it contains labels given by the user

        self._permutation = multiple_permutation.MultiplePermutation(new_perm,alphabet=perm.alphabet(),flips=perm.flips()) 
        self._rotation = rotation

    
    def vector_space(self):
        return self._vector_space

    
    def base_ring(self):
        r"""
        Returns the base ring over which the discontinuity points are defined

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met
            
            sage: p = mp.MultiplePermutation([[['a', 'b', 'c'], ['d', 'b', 'c']], [['d', 'e', 'f'], ['a', 'e', 'f']]])

            sage: T = met.MultipleExchangeTransformation(p, [1, 2, 3, 1, 5, 6], [[3, 12], [1, 2]])
            sage: T.base_ring()
            Integer Ring

            sage: T = met.MultipleExchangeTransformation(p, [1, 2, 3, 1, 5, 6], [[3, 12], [1, 2/5]])
            sage: T.base_ring()
            Rational Field

            sage: T = met.MultipleExchangeTransformation(p, [1, 2/7, 3, 1, 5, 6], [[3, 12], [0, 2]])
            sage: T.base_ring()
            Rational Field

            sage: T = met.MultipleExchangeTransformation(p, [1, 2, 3, 1, 5, 6], [[3, AA(12).sqrt()], [2, 1]])
            sage: T.base_ring()
            Algebraic Real Field

            sage: sqrt2 = QuadraticField(2).gen()
            sage: T = met.MultipleExchangeTransformation(p, [1, 2, 3, 1, 5, 6], [[1, sqrt2], [1/2, 0]])
            sage: T.base_ring()
            Number Field in a ...
        """
        return self._base_ring

    
    def __copy__(self) :
        res = self.__class__()
        res._base_ring = self._base_ring
        res._vector_space = self._vector_space
        res._permutation = copy(self._permutation)
        res._lengths = copy(self._lengths)
        res._rotation = copy(self._rotation)
        return res


    def permutation(self):
        r"""
        Returns the permutation associated to this iet, with the convention that the top and bottom begin with the first complete interval after 0.

        OUTPUT:

        permutation -- the permutation associated to this met

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['d', 'b', 'c']], [['d', 'e', 'f'], ['a', 'e', 'f']]])
            sage: p = met.MultipleExchangeTransformation(perm, [1, 2, 3, 1, 5, 6], [[0, 1/2], [1/4, 1/3]])
            sage: p.permutation() == perm
            True

            sage: p = met.MultipleExchangeTransformation(perm, [1, 2, 3, 1, 5, 6], [[0, 4], [0, 4]])
            sage: p.permutation() == perm
            False
            sage: p.permutation() == mp.MultiplePermutation([[['a', 'b', 'c'], ['c', 'd', 'b']], [['d', 'e', 'f'], ['a', 'e', 'f']]])
            True
            sage: p.permutation() # this time the test is successful!
            [[['a', 'b', 'c'], ['c', 'd', 'b']], [['d', 'e', 'f'], ['a', 'e', 'f']]]
        """
        return copy(self._permutation)


    def components_length(self):
        r"""
        Returns the list of the total length of each component.
        
        OUTPUT:

        list -- the list of the total length of each component.
        
        EXAMPLES::

            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['d', 'b', 'c']], [['d', 'e', 'f'], ['a', 'e', 'f']]])
            
            sage: t = met.MultipleExchangeTransformation(perm, [1, 2, 3, 1, 5, 6], [[0, 1/2], [1/4, 1/3]])
            sage: t.components_length()
            [6, 12]
        """
        shape = self._permutation.shape()
        ncomp = self._permutation.number_of_components()
        res = []
        for k in range(ncomp):
            lab_k = self._permutation._labels[k][0]
            len_comp_k = sum([self._lengths[i] for i in lab_k])
            res.append(len_comp_k)
        return res
    
    def length(self, label=None):
        r"""
        Returns the length of a subinterval (if ``label`` is specified), or the total length if ``label`` is None.

        EXAMPLES::

            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['d', 'b', 'c']], [['d', 'e', 'f'], ['a', 'e', 'f']]])
            
            sage: t = met.MultipleExchangeTransformation(perm, [1, 2, 3, 1, 5, 6], [[0, 1/2], [1/4, 1/3]])
            sage: t.length()
            18
            sage: t.length(label='a')
            1
            sage: t.length(label='e')
            5
        """
        if label is None:
            return sum(self._lengths)
        else :
            i = self._permutation._alphabet.rank(label)
            return self._lengths[i]    
    
            
    def lengths(self):
        r"""
        Returns the list of lengths associated to this cet.

        OUTPUT:

        vector -- the list of lengths of subinterval (the order of the entries
                  correspond to the alphabet)

        EXAMPLES::

            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['d', 'b', 'c']], [['d', 'e', 'f'], ['a', 'e', 'f']]])
            
            sage: t = met.MultipleExchangeTransformation(perm, [1, 2, 3, 1, 5, 6], [[0, 1/2], [1/4, 1/3]])
            sage: t.lengths()
            (1, 2, 3, 1, 5, 6)
        """
        return self._lengths.__copy__()

    
    def rotation(self):
        r"""
        Returns the rotation associated to this iet with the convention that the top and bottom begin with the first complete interval after 0.

        OUTPUT:

        list -- the list of rotation angle in each component of this met
        
        EXAMPLES::

            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['d', 'b', 'c']], [['d', 'e', 'f'], ['a', 'e', 'f']]])
            
            sage: p = met.MultipleExchangeTransformation(perm, [1, 2, 3, 1, 5, 6], [[0, 1/2], [1/4, 1/3]])
            sage: p.rotation()
            [1/2, 1/12]
        """
        return [self._rotation[k][1]-self._rotation[k][0] for k in range(len(self._rotation))]


    def domain_singularities(self):
        r"""
        Returns the list of singularities of T

        OUTPUT:

        list -- positive reals that corresponds to singularities in the top
            interval

        EXAMPLES::

            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['d', 'b', 'c']], [['d', 'e', 'f'], ['a', 'e', 'f']]])
            
            sage: p = met.MultipleExchangeTransformation(perm, [1, 2, 3, 1, 5, 6], [[0, 1/2], [1/4, 1/3]])
            sage: p.domain_singularities()
            [[0, 1, 3], [1/4, 5/4, 25/4]]
        """
        sing = []
        lengths = self.lengths()
        for k in range(self._permutation.number_of_components()):
            l = [self._rotation[k][0]]
            for j in self._permutation._labels[k][0]:
                l.append(l[-1] + lengths[j])
            del(l[-1])
            sing.append(l)
        return sing

    def range_singularities(self):
        r"""
        Returns the list of singularities of `T^{-1}`

        OUTPUT:

        list -- real numbers that are singular for `T^{-1}`


        EXAMPLES::

            sage: import surface_dynamics.interval_exchanges.multiple_permutation as mp
            sage: import surface_dynamics.interval_exchanges.met as met

            sage: perm = mp.MultiplePermutation([[['a', 'b', 'c'], ['d', 'b', 'c']], [['d', 'e', 'f'], ['a', 'e', 'f']]])
            
            sage: p = met.MultipleExchangeTransformation(perm, [1, 2, 3, 1, 5, 6], [[0, 1/2], [1/4, 1/3]])
            sage: p.range_singularities()
            [[1/2, 3/2, 7/2], [1/3, 4/3, 19/3]]
        """
        sing = []
        lengths = self.lengths()
        for k in range(self._permutation.number_of_components()):
            l = [self._rotation[k][1]]
            for j in self._permutation._labels[k][1]:
                l.append(l[-1] + lengths[j])
            del(l[-1])
            sing.append(l)
        return sing


#   à partir d'ici copié collé sans modif   

    def translations(self):
        r"""
        Return the vector of translations operated on each sub interval of the circle.

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: p = iet.Permutation('a b c', 'c b a')
            sage: T = iet.CircleExchangeTransformation(p, [5,1,3], [0, 2])
            sage: T.translations()
            (6, 0, -4)

        The order of the entries correspond to the alphabet::

            sage: p = iet.Permutation('a c d b', 'b d c a', alphabet='abcd')
            sage: T = iet.CircleExchangeTransformation(p, [1, 1, 1, 1], [1/4, 1/2])
            sage: T.translations()
            (13/4, -11/4, 5/4, -3/4)

            sage: p = iet.Permutation('a c d b', 'b d c a')
            sage: T = iet.CircleExchangeTransformation(p, [1, 1, 1, 1], [1/4, 1/2])
            sage: T.translations()
            (13/4, 5/4, -3/4, -11/4)

        """
        #This vector is covariant with respect to the Rauzy matrices::

         #   sage: p = iet.Permutation('a b c d', 'd c b a')
         #   sage: R = p.rauzy_diagram()
         #   sage: g = R.path(p, *'ttbtbtbtbb')
         #   sage: T = g.self_similar_iet()
         #   sage: for i in range(12):
         #   ....:     S, code = T.zorich_move(iterations=i, data=True)
         #   ....:     gg = R.path(p, *code)
         #   ....:     m = gg.matrix()
         #   ....:     assert m * S.lengths() == T.lengths()
         #   ....:     assert m.transpose() * T.translations() == S.translations()
        

        if self._permutation._flips is not None:
            raise TypeError("the cet has flip(s), the function flipped_translations must be used instead")

        dom_sg = self.domain_singularities()
        im_sg = self.range_singularities()

        p = self._permutation._labels
        top_twin = self._permutation._twin[0]
        top = p[0]
        bot = p[1] # question : à quoi sert bot ?

        translations = self.vector_space()()
        for i0,j in enumerate(top):
            i1 = top_twin[i0]
            translations[j] = im_sg[i1] - dom_sg[i0]

        return translations
    
    def flipped_translations(self): 
        r"""
        Return a list [[c_0,...,c_n],(t_0,...,t_n)] such that 
        the iet on the i-th interval is the function x -> c_i*x + t_i
        [c_0,...,c_n] is a list
        (t_0,...,t_n) is a vector
        """
        dom_sg = self.domain_singularities()
        im_sg = self.range_singularities()

        p = self._permutation._labels
        top_twin = self._permutation._twin[0]
        top = p[0]
        bot = p[1]
        if self._permutation._flips is None :
            flips = (len(top))*[1]
        else :
            flips = self._permutation._flips[0]

        translation_parts = self.vector_space()()       
        for i0,j in enumerate(top):
            i1 = top_twin[i0]
            translation_parts[j] = im_sg[i1] - dom_sg[i0]
            #print(f"{i0=},{j=},{translation_parts[j]=}")
            if flips[i0] == -1:
                i0_next = i0 + 1
                if i0_next == len(dom_sg) :
                    i0_next = 0
                translation_parts[j] =  im_sg[i1] + dom_sg[i0_next]

        return [flips,translation_parts]

    def sah_arnoux_fathi_invariant(self): # TODO
        r"""
        Return the Sah-Arnoux-Fathi invariant

        The interval exchange needs to be defined over a number field. The output
        is then a vector with rational entries of dimension `d (d-1) / 2` where
        `d` is the degree of the field.

        EXAMPLES:

        The golden rotation::

            sage: from surface_dynamics import *
            sage: p = iet.Permutation('a b','b a')
            sage: R = p.rauzy_diagram()
            sage: g = R.path(p, 't', 'b')
            sage: T = g.self_similar_iet()
            sage: T.sah_arnoux_fathi_invariant()
            (2)

        The Sah-Arnoux-Fathi invariant is not changed under Rauzy
        (or Zorich) induction::

            sage: S = T.zorich_move(iterations=100)
            sage: S.sah_arnoux_fathi_invariant()
            (2)
            sage: (T.length().n(), S.length().n())
            (2.61803398874989, 0.000000000000000)

        An other rotation::

            sage: g = R.path(p, 't', 'b', 'b')
            sage: T = g.self_similar_iet()
            sage: T.sah_arnoux_fathi_invariant()
            (1)
            sage: T.rauzy_move().sah_arnoux_fathi_invariant()
            (1)

        Arnoux-Yoccoz in genus 3::

            sage: x = polygen(ZZ)
            sage: poly = x^3 - x^2 - x - 1
            sage: l = max(poly.roots(AA, False))
            sage: K.<a> = NumberField(poly, embedding=l)
            sage: top = 'A1l A1r A2 B1 B2 C1 C2'
            sage: bot = 'A1r B2 B1 C2 C1 A2 A1l'
            sage: p = iet.Permutation(top, bot)
            sage: lengths = vector((a+1, a**2-a-1, a**2, a, a, 1, 1))
            sage: T = iet.IntervalExchangeTransformation(p, lengths)
            sage: T.sah_arnoux_fathi_invariant()
            (0, 0, 0)

        Arnoux-Yoccoz examples in genus 4 (an auto-similar iet)::

            sage: x = polygen(ZZ)
            sage: poly = x^4 - x^3 - x^2 - x - 1
            sage: l = max(poly.roots(AA, False))
            sage: K.<a> = NumberField(poly, embedding=l)
            sage: top = 'A1l A1r A2 B1 B2 C1 C2 D1 D2'
            sage: bot = 'A1r B2 B1 C2 C1 D2 D1 A2 A1l'
            sage: p = iet.Permutation(top, bot)
            sage: lengths = vector((a**4-a**3, 2*a**3-a**4, a**3, a**2, a**2, a, a, 1, 1))
            sage: T = iet.IntervalExchangeTransformation(p, lengths)
            sage: T.sah_arnoux_fathi_invariant()
            (0, 0, 0, 0, 0, 0)
            sage: T.zorich_move(iterations=10).sah_arnoux_fathi_invariant()
            (0, 0, 0, 0, 0, 0)

        A cubic example in genus 4 (H_4(6)^hyp)::

            sage: p = iet.Permutation([0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0])
            sage: x = polygen(QQ)
            sage: poly = x^3 - 6*x^2 + 9*x - 3
            sage: emb = AA.polynomial_root(poly, RIF(1.64, 1.66))
            sage: K.<A> = NumberField(poly, embedding=emb)
            sage: lengths= (-4*A^2 + 4*A + 18, -2*A^2 + 19*A - 25, 168*A^2 - 355*A + 128,
            ....:           586*A^2 - 1317*A + 576, -725*A^2 + 1626*A - 707,
            ....:           -11*A^2 - 6*A + 41, -32*A^2 + 100*A - 77, 20*A^2 - 71*A + 63)
            sage: S = iet.IntervalExchangeTransformation(p, lengths)
            sage: S.permutation().stratum_component()
            H_4(6)^hyp
            sage: S.sah_arnoux_fathi_invariant()
            (0, 0, 0)
            sage: S.zorich_move('left', 120).normalize(17) == S
            True

        A quartic example in genus 4 (H_4(6)^even)::

            sage: p = iet.Permutation([0, 1, 2, 3, 4, 5, 6, 7], [7, 1, 0, 5, 4, 3, 2, 6])
            sage: poly = x^4 - 7*x^3 + 14*x^2 - 8*x + 1
            sage: emb = AA.polynomial_root(poly, RIF(0.17, 0.18))
            sage: K.<A> = NumberField(poly, embedding=emb)
            sage: lengths = (2*A^3 - 18*A^2 + 44*A - 4, -10*A^3 + 55*A^2 - 60*A + 10,
            ....:    15*A^3 - 90*A^2 + 120*A - 15, -16*A^3 + 94*A^2 - 132*A + 22,
            ....:    9*A^3 - 56*A^2 + 88*A - 13, 4*A^3 - 16*A^2 + 3*A + 2,
            ....:    -8*A^3 + 57*A^2 - 101*A + 16, 4*A^3 - 26*A^2 + 38*A - 3)
            sage: S = iet.IntervalExchangeTransformation(p, lengths)
            sage: S.permutation().stratum_component()
            H_4(6)^even
            sage: S.sah_arnoux_fathi_invariant()
            (0, 0, 0, 0, 0, 0)
            sage: S.zorich_move('left', 38).normalize(15) == S
            True
        """
        if self.base_ring() is ZZ:
            return free_module_element(ZZ, [])
        elif self.base_ring() is QQ:
            return free_module_element(QQ, [])

        try:
            K, from_V, to_V = self.base_ring().vector_space()
        except (AttributeError, ValueError):
            raise ValueError("the interval exchange needs to be define over a number field")

        return sum(wedge(to_V(u), to_V(v)) for u,v in zip(self.lengths(), self.translations()))

    def normalize(self, total=1, inplace=False):
        r"""
        Returns a circle exchange transformation of normalized lengths.

        The normalization consist in consider a constant homothetic value for
        each lengths in such way that the sum is given (default is 1).

        INPUT:

        - ``total`` - (default: 1) The total length of the interval

        OUTPUT:

        iet -- the normalized iet

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: t = iet.CircleExchangeTransformation(('a b','b a'), [1,3], [1/2, 1/6])
            sage: t.length()
            4
            sage: s = t.normalize(2)
            sage: s.length()
            2
            sage: s.lengths()
            (1/2, 3/2)
            sage: s.rotation()
            -1/6
        """
        try:
            y = float(total)
        except ValueError:
            raise TypeError("unable to convert x (='%s') into a real number"  %(str(x)))

        if total <= 0:
            raise ValueError("the total length must be positive, got {}".format(total))

        if inplace:
            res = self
        else:
            res = copy(self)

        res._rotation[0] *= total / res.length()
        res._rotation[1] *= total / res.length()       
        res._lengths *= total / res.length()
        
        if not inplace:
            return res

    def __repr__(self): #str
        r"""
        A representation string.

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: a = iet.CircleExchangeTransformation(('a','a'), [1], [0, 1/4])
            sage: a   #indirect doctest
            Circle exchange transformation of a circle of perimeter 1 with permutation
            a
            a
        """
        s = "Circle exchange transformation of a circle of perimeter %s "%self.length()
        s += "with permutation\n%s"%self._permutation
        return s

    def is_identity(self):
        r"""
        Returns True if self is the identity.

        OUTPUT:

        boolean -- the answer

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: p = iet.Permutation("a b","b a")
            sage: q = iet.Permutation("c d","d c")
            sage: s = iet.CET(p, [1,5], [0, 1/2])
            sage: t = iet.CET(q, [5,1], [1/2, 0])
            sage: (s*t).is_identity()
            True
            sage: (t*s).is_identity()
            True

            sage: p = iet.Permutation("a","a")
            sage: t = iet.CET(p, [1], [0, 1/4])
            sage: t.is_identity()
            False

            sage: p = iet.Permutation("a","a")
            sage: t = iet.CET(p, [1], [1/4, 1/4])
            sage: t.is_identity()
            True

            sage: p = iet.Permutation("a b","b a")
            sage: t = iet.CET(p, [1,1], [0, 1])
            sage: t.is_identity()
            True

            sage: p = iet.Permutation("a","a", flips=['a'])
            sage: t = iet.CET(p, [1], [1/4, 1/4])
            sage: t.is_identity()
            False
            sage: t2 = t*t
            sage: t2.is_identity()
            True
        """
        return self._permutation.is_identity() and (self._rotation[0]==self._rotation[1]) and (self._permutation._flips is None or self._permutation._flips[1] == [1]*len(self._permutation)) # (self.rotation() == 0) 

    def inverse(self):
        r"""
        Returns the inverse cet.

        OUTPUT:

        cet -- the inverse circle exchange transformation

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: p = iet.Permutation("a b","b a")
            sage: s = iet.CET(p, [1,sqrt(2)-1], [0, 1/4])
            sage: t = s.inverse()
            sage: t.permutation()
            b a
            a b
            sage: t.lengths()
            (1, sqrt(2) - 1)
            sage: t.rotation()
            -1/4
            sage: t*s
            Circle exchange transformation of a circle of perimeter sqrt(2) with permutation
            aa bb
            aa bb

        We can verify with the method .is_identity()::

            sage: p = iet.Permutation("a b c d","d a c b")
            sage: s = iet.CET(p, [1, sqrt(2), sqrt(3), sqrt(5)], [1/5, sqrt(7)])
            sage: (s * s.inverse()).is_identity()
            True
            sage: (s.inverse() * s).is_identity()
            True

            sage: p = iet.Permutation("a b c","c b a",flips=['a'])
            sage: t = iet.CET(p, [4,3,5], [1, 3])
            sage: (t*t.inverse()).is_identity()
            True
            sage: (t.inverse()*t).is_identity()
            True
        """
# question : comment obtenir True pour sqrt(7) - sqrt(2) == sqrt(7) - sqrt(2) ?
        res = copy(self)
        res._permutation = self._permutation.top_bottom_inverse()
        res._rotation = [self._rotation[1],self._rotation[0]]
        return res

    def erase_marked_points(self): # TODO
#sage: from surface_dynamics import *
#            sage: p = iet.Permutation([0,1,2,3,4], [4,2,3,0,1])
#            sage: T1 = iet.CircleExchangeTransformation(p, [13,2,4,5,7], [0,1])
#            sage: T1.erase_marked_points()
#            Circle exchange transformation of a circle of perimeter 40 with permutation
#            0 2 4
#            4 2 0

# BUG : example gives permutation [0, 4], [4, 0]
        """
        Remove the marked points. 

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: p = iet.Permutation([0,1,2,3,4], [4,2,3,0,1])
            sage: T1 = iet.IntervalExchangeTransformation(p, [13,2,4,5,7])
            sage: T1.erase_marked_points()
            Interval exchange transformation of [0, 40[ with permutation
            0 2 4
            4 2 0

        TESTS:

            sage: K.<a> = NumberField(x^5 - 2, embedding=AA(2)**(1/5))

        Left side fake zero::

            sage: p = iet.Permutation([1,5,3,4,6,2], [3,2,5,1,4,6])
            sage: T1 = iet.IntervalExchangeTransformation(p, [1, a, a**2, a**3, a**4, 1 + a])
            sage: T2 = T1.erase_marked_points()
            sage: T2
            Interval exchange transformation of [0, a^4 + a^3 + a^2 + 3*a + 2[ with permutation
            1 3 4 2
            3 2 1 4
            sage: T2.permutation().stratum()
            H_2(2)
            sage: assert T2.length(1) == T1.length(1) + T1.length(5)
            sage: assert T2.length(2) == T1.length(2)
            sage: assert T2.length(3) == T1.length(3) + T1.length(5)
            sage: assert T2.length(4) == T1.length(4) + T1.length(6)
            sage: assert T1.sah_arnoux_fathi_invariant() == T2.sah_arnoux_fathi_invariant()

        Right side fake zero::

            sage: p = iet.Permutation([1,3,4,5,2], [3,2,5,1,4])
            sage: T1 = iet.IntervalExchangeTransformation(p, [1,a,a**2,a**3,a**4])
            sage: T2 = T1.erase_marked_points()
            sage: T2
            Interval exchange transformation of [0, a^4 + 2*a^3 + a^2 + a + 1[ with permutation
            1 3 4 2
            3 2 1 4
            sage: T2.permutation().stratum()
            H_2(2)
            sage: assert T2.length(1) == T1.length(1)
            sage: assert T2.length(2) == T1.length(2) + T1.length(5)
            sage: assert T2.length(3) == T1.length(3)
            sage: assert T2.length(4) == T1.length(4) + T1.length(5)
            sage: assert T1.sah_arnoux_fathi_invariant() == T2.sah_arnoux_fathi_invariant()

        Left and right sides fake zeros::

            sage: p = iet.Permutation([1,5,3,4,6,2], [3,2,6,5,1,4])
            sage: T1 = iet.IntervalExchangeTransformation(p, [1,a+1,a**2+1,a**3,a**4-a,a**3+a+1])
            sage: T2 = T1.erase_marked_points()
            sage: T2
            Interval exchange transformation of [0, 2*a^4 + 2*a^3 + a^2 + a + 5[ with permutation
            1 3 4 2
            3 2 1 4
            sage: T2.permutation().stratum()
            H_2(2)
            sage: assert T2.length(1) == T1.length(1) + T1.length(5)
            sage: assert T2.length(2) == T1.length(2) + T1.length(6)
            sage: assert T2.length(3) == T1.length(3) + T1.length(5)
            sage: assert T2.length(4) == T1.length(4) + T1.length(6)
            sage: assert T1.sah_arnoux_fathi_invariant() == T2.sah_arnoux_fathi_invariant()

        Left-right fake zero::

            sage: p = iet.Permutation([1,3,4,2,5], [2,3,5,1,4])
            sage: T1 = iet.IntervalExchangeTransformation(p, [1, a+1, a**2, a**3, a**4-a])
            sage: T2 = T1.erase_marked_points()
            sage: T2
            Interval exchange transformation of [0, 2*a^4 + 2*a^3 + a^2 - a + 2[ with permutation
            1 3 4 2
            2 3 1 4
            sage: T2.permutation().stratum()
            H_2(2)
            sage: assert T2.length(1) == T1.length(1) + T1.length(5)
            sage: assert T2.length(2) == T1.length(2) + T1.length(5)
            sage: assert T2.length(3) == T1.length(3)
            sage: assert T2.length(4) == T1.length(4) + T1.length(2)
            sage: assert T1.sah_arnoux_fathi_invariant() == T2.sah_arnoux_fathi_invariant()

        A small example that end up wrong::

            sage: p = iet.Permutation([0,2,1,3,4], [4,0,1,2,3])
            sage: K.<a> = NumberField(x^2 - 2, embedding=AA(2)**(1/2))
            sage: lengths = [1, a + 1, a, a + 1, 2*a - 2]
            sage: T = iet.IntervalExchangeTransformation(p, lengths)
            sage: U = T.erase_marked_points()
            sage: U.permutation().stratum()
            H_2(2)
            sage: assert T.sah_arnoux_fathi_invariant() == U.sah_arnoux_fathi_invariant()

            sage: top = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            sage: bot = [15, 11, 12, 9, 10, 7, 8, 5, 6, 2, 3, 4, 16, 14, 0, 1, 13]
            sage: p = iet.Permutation(top, bot)
            sage: x = polygen(QQ)
            sage: poly = x^6 - 6*x^4 + 9*x^2 - 3
            sage: emb = AA.polynomial_root(poly, RIF(1.25, 1.30))
            sage: K.<a> = NumberField(poly, embedding=emb)
            sage: lengths = (-2*a^4 + 7*a^2 - 6, 10/3*a^4 - 17*a^2 + 19, -a^4 + 6*a^2 - 7,
            ....:     -1/3*a^4 + 3*a^2 - 4, -2/3*a^4 + 3*a^2 - 3, 2/3*a^4 + 11*a^2 - 20,
            ....:     2/3*a^4 + 11*a^2 - 20, 8*a^4 - 35*a^2 + 36, 8*a^4 - 35*a^2 + 36,
            ....:     -29/3*a^4 + 42*a^2 - 43, -29/3*a^4 + 42*a^2 - 43, 8/3*a^4 - 28*a^2 + 39,
            ....:     8/3*a^4 - 28*a^2 + 39, -4*a^4 + 20*a^2 - 22, 4/3*a^4 - 4*a^2 + 3,
            ....:     -8/3*a^4 + 16*a^2 - 19, 8/3*a^4 - 14*a^2 + 16)
            sage: T = iet.IntervalExchangeTransformation(p, lengths)
            sage: T.permutation().stratum_component()
            H_4(6, 0^9)^hyp
            sage: T.sah_arnoux_fathi_invariant()
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            sage: U = T.erase_marked_points()
            sage: U.permutation().stratum_component()
            H_4(6)^hyp
            sage: U.sah_arnoux_fathi_invariant()
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        """
        p = self._permutation.__copy__()
        n = len(p)
        lengths = self._lengths
        tt, tb = self._permutation._twin
        lt, lb = self._permutation._labels
        newlengths = list(self._lengths)
        assert max(lt) < len(newlengths)
        assert max(lb) < len(newlengths)
        remove = set()
        newtop = []
        i = 0

        # singularities strictly in the middle
        while i < n:
            k = 1
            while i + k < n and tt[i] + k == tt[i + k]:
                k += 1
            newlengths[lt[i]] = sum(lengths[j] for j in range(i, i+k))
            for j in range(i+1, i+k):
                remove.add(lt[j])
            newtop.append(lt[i])
            i += k

        if remove:
            newbot = [j for j in lb if j not in remove]
            assert set(newtop) == set(newbot)
            p._labels = [newtop, newbot]
            p._init_twin(p._labels)
            tt, tb = p._twin
            lt, lb = newtop, newbot
            n = len(lt)

        i0 = tb[0] - 1
        i1 = tt[0] - 1
        if len(newtop) > 2 and i0 > 0 and i1 > 0 and tt[i0] == i1:
            # left end fake zero
            # the permutation looks like
            # A ... C B ...
            # B ... C A ...
            # backward top-left
            # A ... C B ...   l(A) <- l(A) + l(C)
            # C B ... A ...
            # fusion C in B
            # A ... B ...
            # B ... A ...
            assert lt[i0] == lb[i1]
            A = lt[0]
            B = lb[0]
            C = lt[i0]
            newlengths[A] += newlengths[C]
            newlengths[B] += newlengths[C]
            newlengths[C] = 0
            newtop.pop(newtop.index(C))

            newbot = [i for i in lb if i != C]
            p._labels = [newtop, newbot]
            assert set(newtop) == set(newbot)
            p._init_twin(p._labels)
            tt, tb = p._twin
            lt, lb = newtop, newbot
            n = len(lt)

        i0 = tb[-1] + 1
        i1 = tt[-1] + 1
        if len(newtop) > 2 and i0 < n and i1 < n and tt[i0] == i1:
            # right end fake zero
            # the permutation looks like
            # ... B C ... A
            # ... A C ... B
            # backward top-right
            # ... B C ... A
            # ... A ... B C
            # fusion C in B
            # ... B ... A
            # ... A ... B
            assert lt[i0] == lb[i1]
            A = lt[-1]
            B = lb[-1]
            C = lt[i0]
            assert newtop[-1] == A
            newlengths[A] += newlengths[C]
            newlengths[B] += newlengths[C]
            newlengths[C] = 0
            newtop.pop(newtop.index(C))

            newbot = [i for i in lb if i != C]
            assert set(newtop) == set(newbot)
            p._labels = [newtop, newbot]
            p._init_twin(p._labels)
            tt, tb = p._twin
            lt, lb = newtop, newbot
            n = len(lt)

        i0 = tb[0] - 1 # D
        i1 = tt[0] - 1 # C
        if len(newtop) > 2 and tb[i1] == tt[i0] == n-1:
            # left-right end fake zero
            # the permutation looks like
            # A ... D B ... C  or  A ... D B  or  A B ... C
            # B ... C A ... D      B A ... D      B ... C A
            #                      (B=C case)     (A=D case)
            assert lb[-1] == lt[i0] and lt[-1] == lb[i1]
            A = lt[0]
            B = lb[0]
            C = lb[i1]
            D = lt[i0]
            assert newtop[0] == A and newtop[-1] == C and B != D
            assert A in newtop and B in newtop and C in newtop and D in newtop
            if B == C:
                # A ... D B
                # B A ... D
                # backward bot-left
                # D A ... B    l(B) <- l(B) + l(D)
                # B A ... D
                # backward top-right
                # D A ... B    l(B) <- l(B) + l(A)
                # B ... D A
                # fusion A in D
                # D ... B
                # B ... D
                assert newtop[0] == A and newtop[-1] == B and newtop[-2] == D
                newlengths[B] += newlengths[A] + newlengths[D]
                newlengths[D] += newlengths[A]
                newlengths[A] = 0
                newtop.pop(0)
                newtop.pop(-2)
                newtop.insert(0, D)
                assert newtop[0] == D and newtop[-1] == B

                newbot = [i for i in lb if i != A]

            elif A == D:
                # A B ... C
                # B ... C A
                # backward bot-right
                # A ... C B   l(A) <- l(A) + l(B)
                # B ... C A
                # backward top-left
                # A ... C B   l(A) <- l(A) + l(C)
                # C B ... A
                # fusion C in B
                # A ... B
                # B ... A
                newlengths[A] += newlengths[B] + newlengths[C]
                newlengths[B] += newlengths[C]
                newlengths[C] = 0
                assert newtop[-1] == C
                assert newtop[1] == B
                newtop.pop(-1)
                newtop.pop(1)
                newtop.append(B)
                assert newtop[0] == A and newtop[-1] == B, (newtop, A, B)
                assert C not in newtop
                newbot = [i for i in lb if i != C]

            else:
                # do backward Rauzy top-left / bottom-right
                # A ... D ... C B     l(A) <- l(A) + l(C)
                # C B ... A ... D     l(D) <- l(D) + l(B)
                # fusion C in B
                # A ... D ... B
                # B ... A ... D
                newlengths[A] += newlengths[C]  # backward rauzy
                newlengths[D] += newlengths[B]  # backward rauzy
                newlengths[B] += newlengths[C]  # absorption
                newlengths[C] = 0
                newtop.pop(newtop.index(C))
                newtop.pop(newtop.index(B))
                newtop.append(B)
                newbot = [i for i in lb if i != C]

            assert set(newtop) == set(newbot)
            p._labels = [newtop, newbot]
            p._init_twin(p._labels)
            tt, tb = p._twin
            lt, lb = newtop, newbot

        unrank = self._permutation.alphabet().unrank
        newtop = [unrank(lab) for lab in lt]
        newbot = [unrank(lab) for lab in lb]
        newlengths = [newlengths[j] for j in lt]
        p = LabelledPermutationIET([newtop, newbot])
        return CircleExchangeTransformation(p, newlengths)

    def __mul__(self, other):
        r"""
        Composition of iet.

        The domain (i.e. the length) of the two iets must be the same. The
        alphabet chosen depends on the permutation.

        TESTS:

            sage: from surface_dynamics import *

        ::

            sage: p = iet.Permutation("a b", "a b")
            sage: t = iet.CET(p, [1,1], [0, 0])
            sage: r = t*t
            sage: r.permutation()
            aa bb
            aa bb
            sage: r.lengths()
            (1, 1)

        ::

            sage: p = iet.Permutation("a b","b a")
            sage: t = iet.CET(p, [1,1], [0, 0])
            sage: r = t*t
            sage: r.permutation()
            ab ba
            ab ba
            sage: r.lengths()
            (1, 1)

        ::

            sage: p = iet.Permutation("a b", "a b")
            sage: t = iet.CET(p, [1,1], [0, 1/4])
            sage: r = t*t
            sage: r.permutation()
            aa ab bb ba
            ba aa ab bb
            sage: r.lengths()
            (3/4, 1/4, 3/4, 1/4)
            sage: r.rotation()
            1/4

        ::

            sage: p1 = iet.Permutation("a b c", "b a c", flips=['b'])
            sage: t1 = iet.CET(p1, [4,3,5], [1, 4])
            sage: p2 = iet.Permutation("d e f g", "g f e d", flips=['e','g'])
            sage: t2 = iet.CET(p2, [3,4,2,3], [2, 3])
            sage: r = t2*t1
            sage: r.permutation()
            -ae  af  be -bd -cg  cd
             cd -bd -cg  af -ae  be
            sage: r.lengths()
            (2, 2, 2, 1, 3, 2)
            sage: r._rotation
            [1, 0]

        :: #TODO

            sage: p = iet.Permutation("1 2 3 4 5","5 4 3 2 1")
            sage: q = iet.Permutation("a b","b a")
            sage: s = iet.IET(p, [1]*5)
            sage: t = iet.IET(q, [1/2, 9/2])
            sage: r = s*t
            sage: r.permutation()
            a5 b1 b2 b3 b4 b5
            b5 a5 b4 b3 b2 b1
            sage: r.lengths()
            (1/2, 1, 1, 1, 1, 1/2)
            sage: r = t*s
            sage: r.permutation()
            1b 2b 3b 4b 5a 5b
            5b 4b 3b 2b 1b 5a
            sage: r.lengths()
            (1, 1, 1, 1, 1/2, 1/2)
            sage: t = iet.IET(q, [3/2, 7/2])
            sage: r = s*t
            sage: r.permutation()
            a4 a5 b1 b2 b3 b4
            a5 b4 a4 b3 b2 b1
            sage: r.lengths()
            (1/2, 1, 1, 1, 1, 1/2)
            sage: t = iet.IET(q, [5/2,5/2])
            sage: r = s*t
            sage: r.permutation()
            a3 a4 a5 b1 b2 b3
            a5 a4 b3 a3 b2 b1
            sage: r = t*s
            sage: r.permutation()
            1b 2b 3a 3b 4a 5a
            3b 2b 1b 5a 4a 3a

        ::

            sage: p = iet.Permutation("a b","b a")
            sage: s = iet.IET(p, [4,2])
            sage: q = iet.Permutation("c d","d c")
            sage: t = iet.IET(q, [3, 3])
            sage: r1 = t * s
            sage: r1.permutation()
            ac ad bc
            ad bc ac
            sage: r1.lengths()
            (1, 3, 2)
            sage: r2 = s * t
            sage: r2.permutation()
            ca cb da
            cb da ca
            sage: r2.lengths()
            (1, 2, 3)

        ::

            sage: p = iet.Permutation("a b c","c b a",flips=['a'])
            sage: t = iet.CET(p, [4,3,5], [1, 3])
            sage: t2 = t*t
            sage: t2.permutation()
            aa -ac bc -ca cb
            aa bc -ac cb -ca
            sage: t2.lengths()
            (2, 2, 3, 2, 3)
            sage: t2.flipped_translations()
            [[1, -1, 1, -1, 1], (0, 11, -2, 21, -2)]
            sage: t2._rotation
            [1, 1]

        ::

            sage: p = iet.Permutation("a b c d","d c b a",flips=['b','c'])
            sage: t = iet.IET(p, [1/4,3/20,1/2,1/10])
            sage: t2 = t*t
            sage: t2.permutation()
            -ac ad bc cc cb -ca da
            ad -ac bc cc cb da -ca
            sage: t2.lengths()
            (3/20, 1/10, 3/20, 1/5, 3/20, 3/20, 1/10)
            sage: t8 = t2*t2*t2*t2
            sage: t8.is_identity()
            True
            sage: t8.permutation()
            acadacad acacacac adacadac bcbcbcbc cccccccc cbcbcbcb cadacada cacacaca dacadaca
            acadacad acacacac adacadac bcbcbcbc cccccccc cbcbcbcb cadacada cacacaca dacadaca
            sage: t8.lengths()
            (1/10, 1/20, 1/10, 3/20, 1/5, 3/20, 1/10, 1/20, 1/10)

        ::

            sage: p = iet.Permutation("a b c d","d b c a",flips=['b','c'])
            sage: r = iet.IET(p, [4/10,1/10,3/10,2/10])
            sage: r2 = r*r
            sage: r2.permutation()
            -ac ad -ba cc cb -ca da
            ad cb -ac cc da -ba -ca
            sage: r2.lengths()
            (1/5, 1/5, 1/10, 1/10, 1/10, 1/10, 1/5)

        ::
        
            sage:

        """
        assert(
            isinstance(other, CircleExchangeTransformation) and
            self.length() == other.length())

        from .labelled import LabelledPermutationIET, FlippedLabelledPermutationIET
        
        ### Initialization
        other_sg = other.range_singularities() #[1:]
        self_sg = self.domain_singularities() #[1:]
        
        n_other = len(other._permutation)
        n_self = len(self._permutation)
        
        r_other = other._rotation[1]

        total_length = self.length()
        interval_other = other._permutation._labels[1]
        interval_self = self._permutation._labels[0]

        if other._permutation._flips is None:
            flips_other0 = [1]*n_other
            flips_other1 = [1]*n_other
        else:
            flips_other0 = other._permutation._flips[0]
            flips_other1 = other._permutation._flips[1]
        if self._permutation._flips is None:
            flips_self0 = [1]*n_self
            flips_self1 = [1]*n_self
        else:
            flips_self0 = self._permutation._flips[0]
            flips_self1 = self._permutation._flips[1]


        d_other = dict([(i, []) for i in interval_other])
        d_self = dict([(i, []) for i in interval_self])

        i_other = 0
        i_self = 0
        
        #print("other_sg (bot)", other_sg)
        #print("self_sg (top)", self_sg)
        #print("n_other", n_other)
        #print("n_self", n_self)
        #print("r_other", r_other)
        #print("interval_other", interval_other)
        #print("interval_self", interval_self)
        
        while i_self < n_self and r_other >= self_sg[i_self] : # if i_self > n_self, it means that rotation of other is in the last interval of self
            i_self += 1

        i_self -= 1
        i_self_start = i_self
        #print("i_self (1e interval à intersecter le 1e de other)", i_self)

        x = r_other
        l_lengths = []
        
        ### build dictonaries d_other and d_self containing the intervals of continuity "that plays together" in the composition, and the lengths of intervals of continuity of the composition
        # i_other and i_self are the indices of current composed intervals
        while i_other < n_other and i_self < n_self:
            j_other = interval_other[i_other]
            flip_other = flips_other1[i_other]
            j_self = interval_self[i_self]
            flip_self = flips_self0[i_self]
            #print("j_other", j_other)
            #print("flip_other", flip_other)
            #print("j_self", j_self)
            #print("flip_self", flip_self)

            d_other[j_other].append((j_self,flip_self))
            d_self[j_self].append((j_other,flip_other))
            
            if i_other == n_other - 1 :
                other_next_sg = other_sg[0] + total_length
            else:
                other_next_sg = other_sg[i_other+1]

            if i_self == n_self - 1 :
                self_next_sg = self_sg[0] + total_length
            else:
                self_next_sg = self_sg[i_self+1]

            if other_next_sg < self_next_sg:
                l = other_next_sg - x
                x = other_next_sg
                i_other += 1
            elif other_next_sg > self_next_sg:
                l = self_next_sg - x
                x = self_next_sg
                i_self += 1
            else:
                l = self_next_sg - x
                x = self_next_sg
                i_other += 1
                i_self += 1
            l_lengths.append(((j_other,j_self),l))

        i_self = 0
        c = 0 # counts the number of time we have seen the interval interval_self[i_self]

        while i_other < n_other and i_self <= i_self_start:
            j_other = interval_other[i_other]
            flip_other = flips_other1[i_other]
            j_self = interval_self[i_self]
            flip_self = flips_self0[i_self]

            d_other[j_other].append((j_self,flip_self))
            d_self[j_self].insert(c, (j_other,flip_other))

            if i_other == n_other - 1 :
                other_next_sg = other_sg[0] + total_length
            else:
                other_next_sg = other_sg[i_other+1]

            if i_self == n_self - 1 : # should never be satisfied. which case gives True here ? one interval or ... ? i_self_start == n_self is not possible at most n_self-1
                self_next_sg = self_sg[0] + total_length
            else:
                self_next_sg = self_sg[i_self+1] + total_length

            if other_next_sg < self_next_sg:
                l = other_next_sg - x
                x = other_next_sg
                i_other += 1
                c += 1
            elif other_next_sg > self_next_sg:
                l = self_next_sg - x
                x = self_next_sg
                i_self += 1
                c = 0
            else:
                l = self_next_sg - x
                x = self_next_sg
                i_other += 1
                i_self += 1
                c = 0

            l_lengths.append(((j_other,j_self),l))

        alphabet_other = other._permutation.alphabet()
        alphabet_self = self._permutation.alphabet()

        d_lengths = dict(l_lengths)

        l_lengths = []
        top_interval = []
        flips = []
        for i,i_flip in zip(other._permutation._labels[0],flips_other0):
            sub_interval = []
            sub_interval_lengths = []
            for j,j_flip in d_other[i]:
                a = alphabet_other.unrank(i)
                b = alphabet_self.unrank(j)
                sub_interval.append(str(a)+str(b))
                sub_interval_lengths.append(d_lengths[(i,j)])
                if j_flip*i_flip == -1 :   # the new interval of continuity is flipped if and only if exactly one of interval number i from other and number j from self is flip
                    flips.append(str(a)+str(b))
            if i_flip == -1 :
                sub_interval.reverse()
                sub_interval_lengths.reverse()
            top_interval += sub_interval
            l_lengths += sub_interval_lengths

        bottom_interval = []
        for i,i_flip in zip(self._permutation._labels[1],flips_self1):
            sub_interval = []
            for j,j_flip in d_self[i]:
                a = alphabet_other.unrank(j)
                b = alphabet_self.unrank(i)
                sub_interval.append(str(a)+str(b))
            if i_flip == -1 :
                sub_interval.reverse()
            bottom_interval += sub_interval
        
        if self._permutation._flips is None and other._permutation._flips is None:
            p = LabelledPermutationIET((top_interval,bottom_interval))
        else :
            p = FlippedLabelledPermutationIET((top_interval,bottom_interval),flips=flips)

        return CircleExchangeTransformation(p,l_lengths,[other._rotation[0],self._rotation[1]])



    def __eq__(self, other):
        r"""
        Tests equality

        TESTS::

            sage: from surface_dynamics import *

            sage: t = iet.IntervalExchangeTransformation(('a b','b a'),[1,1])
            sage: t == t
            True

            sage: perm = iet.Permutation('a b c','c b a')
            sage: t1 = iet.CircleExchangeTransformation(perm,(1,2,3), [1/4, 1/3])
            sage: perm = iet.Permutation('b c a','c b a')
            sage: t2 = iet.CircleExchangeTransformation(perm,(2,3,1), [5/4, 1/3])
            sage: t1 == t2
            True
        """
        return (
            isinstance(self, type(other)) and
            self._permutation == other._permutation and
            self._lengths == other._lengths and
            self._rotation == other._rotation) # question : on devrait en plus dire qu'il y a égalité quand les perm et rotation sont différentes mais donnent la même chose en terme de fonction - devrait marcher, à vérifier

    def __ne__(self, other):
        r"""
        Tests difference

        TESTS::

            sage: from surface_dynamics import *

            sage: t = iet.IntervalExchangeTransformation(('a b','b a'),[1,1])
            sage: t != t
            False

            sage: perm = iet.Permutation('a b c','c b a')
            sage: t1 = iet.CircleExchangeTransformation(perm,(1,2,3), [1/4, 1/3])
            sage: perm = iet.Permutation('b c a','c b a')
            sage: t2 = iet.CircleExchangeTransformation(perm,(2,3,1), [5/4, 1/3])
            sage: t1 != t2
            False

            sage: t != t1
            True
        """
        return (
            not isinstance(self, type(other)) or
            self._permutation != other._permutation or
            self._lengths != other._lengths or
            self._rotation != other._rotation) # question : on devrait en plus dire qu'il y a égalité quand les perm et rotation sont différentes mais donnent la même chose en terme de fonction - devrait marcher, à vérifier

    def recoding(self, n): # TODO
        r"""
        Recode this circle exchange transformation on the words of length
        ``n``.

        EXAMPLES::

            sage: from surface_dynamics import *
            sage: p = iet.Permutation('a d c b', 'b c a d', alphabet='abcd')
            sage: T = iet.IntervalExchangeTransformation(p, [119,213,82,33])
            sage: T.recoding(2)
            Interval exchange transformation of [0, 447[ with permutation
            ab db cc cb ba bd bc
            ba bd bc cc cb ab db
            sage: T.recoding(3)
            Interval exchange transformation of [0, 447[ with permutation
            aba abd abc dbc ccb cba bab bdb bcc bcb
            cba aba abd abc dbc bcc bcb ccb bab bdb
        """
        length = len(self._permutation)
        lengths = self._lengths
        A = self.permutation().alphabet()
        unrank = A.unrank

        top = self._permutation._labels[0]
        bot = self._permutation._labels[1]
        bot_twin = self._permutation._twin[1]

        sg_top = self.domain_singularities()[1:-1]  # iterates of the top singularities
        cuts = [[[i], lengths[i]] for i in bot]     # the refined bottom interval

        translations = self.translations()

        for step in range(n-1):
            i = 0
            y = self.base_ring().zero()
            new_sg_top = []
            limits = [0]
            for j,x in enumerate(sg_top):
                while y < x:
                    cuts[i][0].append(top[j])
                    y += cuts[i][1]
                    i += 1

                limits.append(i)
                if y != x:
                    cuts.insert(i, [cuts[i-1][0][:-1], cuts[i-1][1]])
                    cuts[i-1][1] -= y-x
                    cuts[i][0].append(top[j+1])
                    cuts[i][1] = y-x
                    i += 1
            while i < len(cuts):
                cuts[i][0].append(top[j+1])
                i += 1
            limits.append(len(cuts))

            # now we reorder cuts with respect to T according to the cut at
            # limits
            if step != n-2:
                new_cuts = []
                for j in bot_twin:
                    new_cuts.extend(cuts[limits[j]:limits[j+1]])
                cuts = new_cuts

        # build the new interval exchange transformations
        itop = [None]*(max(top)+1)
        for i,j in enumerate(top):
            itop[j] = i

        def key(x):
            return [itop[i] for i in x[0]]
        top = sorted(cuts, key=key)
        lengths = [y for x,y in top]
        top = [''.join(str(unrank(i)) for i in x) for x,y in top]
        bot = cuts
        bot = [''.join(str(unrank(i)) for i in x) for x,y in bot]

        from .labelled import LabelledPermutationIET
        p = LabelledPermutationIET((top,bot))
        return CircleExchangeTransformation(p,lengths)

    def in_which_interval(self, x, interval=0):
        r"""
        Returns the letter for which x is in this interval.

        INPUT:

        - ``x`` - a positive number

        - ``interval`` - (default: 'top') 'top' or 'bottom'


        OUTPUT:

        label -- a label corresponding to an interval

        TESTS::

            sage: from surface_dynamics import *

            sage: t = iet.CircleExchangeTransformation(('a b c','c b a'),[1,1,1],[2/10,1/2])
            sage: t.in_which_interval(0)
            'c'
            sage: t.in_which_interval(0.3)
            'a'
            sage: t.in_which_interval(1)
            'a'
            sage: t.in_which_interval(1.9)
            'b'
            sage: t.in_which_interval(2.2)
            'c'
            sage: t.in_which_interval(2.1)
            'b'
            sage: t.in_which_interval(3)
            Traceback (most recent call last):
            ...
            ValueError: your value does not lie in [0; 3[

        .. and for the bottom interval::

            sage: t.in_which_interval(0,'bottom')
            'a'
            sage: t.in_which_interval(1.2,'bottom')
            'c'
            sage: t.in_which_interval(2.4,'bottom')
            'b'
        """
        interval = interval_conversion(interval)

        if x < 0 or x >= self.length():
            raise ValueError("your value does not lie in [0; {}[".format(self.length()))

        i = 0
        x -= self._rotation[interval]
        if x < 0 :
            i = len(self._permutation)

        while x >= 0:
            x -= self._lengths[self._permutation._labels[interval][i]]
            i += 1

        i -= 1
        x += self._lengths[self._permutation._labels[interval][i]]

        j = self._permutation._labels[interval][i]
        return self._permutation._alphabet.unrank(j)

    def singularities(self):
        r"""
        The list of singularities of 'T' and 'T^{-1}'.

        OUTPUT:

        list -- two lists of positive numbers which corresponds to extremities
            of subintervals

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: t = iet.CircleExchangeTransformation(('a b','b a'), [1/2,3/2], [0, 1/4])
            sage: t.singularities()
            [[0, 1/2], [1/4, 7/4]]
        """
# Attention, on a supprimé la dernière singularité (une singularité de moins que pour les intervalles), qui correspond à la première !
        return [self.domain_singularities(), self.range_singularities()]


    def __call__(self, value):
        r"""
        Return the image of value by this transformation

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: t = iet.CircleExchangeTransformation(('a b','b a'), [1/2, 3/2], [2/10, 4/10])
            sage: t(0)
            17/10
            sage: t(7/10)
            2/5
            sage: t(3/10)
            0
            sage: t(2/10)
            19/10
        """
        assert(value >= 0 and value < self.length())

        dom_sg = self.domain_singularities()
        im_sg = self.range_singularities()

        a = self.in_which_interval(value)

        i0 = self._permutation[0].index(a)  # Attention, TODO? : gérer les décalages d'indice !
        i1 = self._permutation[1].index(a)

        res = value - dom_sg[i0] + im_sg[i1]
        
        if self._permutation._flips is not None and self._permutation._flips[0][i0] == -1 :
            res = -value + dom_sg[i0+1] + im_sg[i1]
        
        while res < 0 :
            res += self.length()
        while res >= self.length() :
            res -= self.length()

        return res

# TODO
    def rauzy_move(self, side='right', iterations=1, data=False, error_on_saddles=True):
        r"""
        Performs a Rauzy move.

        INPUT:

        - ``side`` - 'left' (or 'l' or 0) or 'right' (or 'r' or 1)

        - ``iterations`` - integer (default :1) the number of iteration of Rauzy
           moves to perform

        - ``data`` - whether to return also the paths and composition of towers

        - ``error_on_saddles`` - (default: ``True``) whether to stop when a saddle
          is encountered

        OUTPUT:

        - ``iet`` -- the Rauzy move of self

        - ``path`` -- (if ``data=True``) a list of 't' and 'b'

        - ``towers`` -- (if ``data=True``) the towers of the Rauzy induction as a word morphism

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: phi = QQbar((sqrt(5)-1)/2)
            sage: t1 = iet.IntervalExchangeTransformation(('a b','b a'),[1,phi])
            sage: t2 = t1.rauzy_move().normalize(t1.length())
            sage: l2 = t2.lengths()
            sage: l1 = t1.lengths()
            sage: l2[0] == l1[1] and l2[1] == l1[0]
            True

            sage: tt,path,sub = t1.rauzy_move(iterations=3, data=True)
            sage: tt
            Interval exchange transformation of [0, 0.3819660112501051?[ with
            permutation
            a b
            b a
            sage: path
            ['b', 't', 'b']
            sage: sub
            WordMorphism: a->aab, b->aabab

        The substitution can also be recovered from the Rauzy diagram::

            sage: p = t1.permutation()
            sage: p.rauzy_diagram().path(p, *path).substitution() == sub
            True

        An other examples involving 3 intervals::

            sage: t = iet.IntervalExchangeTransformation(('a b c','c b a'),[1,1,3])
            sage: t
            Interval exchange transformation of [0, 5[ with permutation
            a b c
            c b a
            sage: t1 = t.rauzy_move()
            sage: t1
            Interval exchange transformation of [0, 4[ with permutation
            a b c
            c a b
            sage: t2 = t1.rauzy_move()
            sage: t2
            Interval exchange transformation of [0, 3[ with permutation
            a b c
            c b a
            sage: t2.rauzy_move()
            Traceback (most recent call last):
            ...
            ValueError: saddle connection found
            sage: t2.rauzy_move(error_on_saddles=False)
            Interval exchange transformation of [0, 2[ with permutation
            a b
            a b

        Degenerate cases::

            sage: p = iet.Permutation('a b', 'b a')
            sage: T = iet.IntervalExchangeTransformation(p, [1,1])
            sage: T.rauzy_move(error_on_saddles=False)
            Interval exchange transformation of [0, 1[ with permutation
            a
            a
        """
        if data:
            towers = {a:[a] for a in self._permutation.letters()}
            path = []

        side = side_conversion(side)

        res = copy(self)
        for i in range(iterations):
            winner,(a,b,c) = res._rauzy_move(side, error_on_saddles=error_on_saddles)
            if data:
                if winner is None:
                    raise ValueError("does not handle degenerate situations")
                towers[a] = towers[b] + towers[c]
                path.append(winner)

        if data:
            from sage.combinat.words.morphism import WordMorphism
            return res, path, WordMorphism(towers)
        else:
            return res

    def backward_rauzy_move(self, winner, side='right'):
        r"""
        Return a new interval exchange transformation obtained by performing a backward Rauzy move.

        EXAMPLES::

            sage: from surface_dynamics import iet
            sage: perm = iet.Permutation('a b c d e f', 'f c b e d a')
            sage: x = polygen(QQ)
            sage: poly = x^3 - x^2 - x - 1
            sage: root = AA.polynomial_root(poly, RIF(1.8, 1.9))
            sage: K.<a> = NumberField(poly, embedding=root)
            sage: T = iet.IntervalExchangeTransformation(perm, [a**2, a-1, a + 2, 3, 2*a - 3, 1])
            sage: T
            Interval exchange transformation of [0, a^2 + 4*a + 2[ with permutation
            a b c d e f
            f c b e d a
            sage: S = T.backward_rauzy_move(0).backward_rauzy_move(1).backward_rauzy_move(1)
            sage: S
            Interval exchange transformation of [0, a^2 + 7*a + 4[ with permutation
            a b c f d e
            f b e d a c
            sage: S.rauzy_move(iterations=3) == T
            True
        """
        winner = interval_conversion(winner)
        side = side_conversion(side)
        res = copy(self)
        res._backward_rauzy_move(winner, side)
        return res

    def zorich_move(self, side='right', iterations=1, data=False):
        r"""
        Performs a Rauzy move.

        INPUT:

        - ``side`` - 'left' (or 'l' or 0) or 'right' (or 'r' or 1)

        - ``iterations`` - integer (default :1) the number of iteration of Rauzy
           moves to perform

        - ``data`` - whether to return also the path

        OUTPUT:

        - ``iet`` -- the Rauzy move of self

        - ``path`` -- (if ``data=True``) a list of 't' and 'b'

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: p = iet.Permutation('a b c', 'c b a')
            sage: T = iet.IntervalExchangeTransformation(p, [12, 35, 67])
            sage: T.zorich_move()
            Interval exchange transformation of [0, 55[ with permutation
            a b c
            c a b
            sage: assert T.permutation() == p and T.lengths() == vector((12,35,67))

        A self similar example in genus 2::

            sage: p = iet.Permutation('a b c d', 'd a c b')
            sage: R = p.rauzy_diagram()
            sage: code = 'b'*4 + 't'*1 + 'b'*3 + 't'*1 + 'b'*3 + 't'*1 + 'b'*1 + 't'*1 + 'b'*4 + 't'*1 + 'b'*2 + 't'*7
            sage: g = R.path(p, *code)
            sage: m = g.matrix()
            sage: poly = m.charpoly()
            sage: l = max(poly.roots(AA, False))
            sage: K.<a> = NumberField(poly, embedding=l)
            sage: lengths = (m - a).right_kernel().basis()[0]
            sage: T = iet.IntervalExchangeTransformation(p, lengths)
            sage: T.normalize(a, inplace=True)
            sage: T
            Interval exchange transformation of [0, a[ with permutation
            a b c d
            d a c b
            sage: T2, path = T.zorich_move(iterations=12, data=True)
            sage: a*T2.lengths() == T.lengths()
            True
            sage: path == code
            True

        Saddle connection detection::

            sage: p = iet.Permutation('a b c', 'c b a')
            sage: T = iet.IntervalExchangeTransformation(p, [41, 22, 135])
            sage: T.zorich_move(iterations=100)
            Traceback (most recent call last):
            ...
            ValueError: saddle connection found
            sage: p = iet.Permutation('a b c d e f', 'f c b e d a')
            sage: T = iet.IntervalExchangeTransformation(p, [41, 132, 22, 135, 55, 333])
            sage: T.zorich_move(iterations=100)
            Traceback (most recent call last):
            ...
            ValueError: saddle connection found
        """
        if data:
            path = ''

        side = side_conversion(side)

        res = copy(self)
        for i in range(iterations):
            winner, m = res._zorich_move(side)
            if data:
                path += winner * m

        if data:
            return res, path
        else:
            return res

    def _rauzy_move(self, side=-1, error_on_saddles=True):
        r"""
        Perform a Rauzy move inplace

        INPUT:

        - ``side`` - must be 0 or -1 (no verification)

        - ``error_on_saddles`` - (default ``True``) whether an error is raised
          when a saddle connections is encountered

        OUTPUT:

        - ``T`` - the iet after Rauzy induction

        - ``winner`` - either ``'t'`` or ``'b'`` (and possibly ``None`` if
          ``error_on_saddles`` is ``False``)

        - ``(a,b,c)`` - a triple of letter such that the towers are obtained by
          applying the substitution `a \mapsto bc`.

        TESTS::

            sage: from surface_dynamics import *

            sage: p = iet.Permutation('a b c d', 'd c b a')
            sage: K.<sqrt2> = QuadraticField(2)
            sage: T = iet.IntervalExchangeTransformation(p, [1,1,1,sqrt2])
            sage: T._rauzy_move()
            ('t', ('a', 'a', 'd'))
            sage: T._rauzy_move()
            ('b', ('d', 'b', 'd'))
            sage: T._rauzy_move()
            ('t', ('b', 'b', 'c'))
            sage: T._rauzy_move()
            ('b', ('c', 'b', 'c'))
            sage: T._rauzy_move()
            ('t', ('b', 'b', 'd'))
            sage: T._rauzy_move()
            ('b', ('d', 'c', 'd'))
            sage: T._rauzy_move()
            ('t', ('c', 'c', 'd'))
            sage: T._rauzy_move()
            ('b', ('d', 'a', 'd'))
            sage: T
            Interval exchange transformation of [0, -4*sqrt2 + 7[ with permutation
            a d b c
            d c b a

        Saddle connections::

            sage: p = iet.Permutation('a b c', 'c b a')
            sage: T = iet.IntervalExchangeTransformation(p, [1,2,1])
            sage: T._rauzy_move()
            Traceback (most recent call last):
            ...
            ValueError: saddle connection found
            sage: T._rauzy_move(error_on_saddles=False)
            (None, ('a', 'a', 'c'))
            sage: T.lengths()
            (1, 2, 0)
            sage: T.permutation()
            a b
            a b

            sage: p = iet.Permutation([0, 1, 2, 3], [3, 1, 2, 0])
            sage: T = iet.IntervalExchangeTransformation(p, [13, 2, 22, 13])
            sage: T._rauzy_move(0, error_on_saddles=False)
            (None, (3, 3, 0))
            sage: T
            Interval exchange transformation of [0, 37[ with permutation
            1 2 3
            1 2 3
            sage: print(T._lengths)
            (0, 2, 22, 13)
            sage: print(T._permutation._labels)
            [[1, 2, 3], [1, 2, 3]]
        """
        top = self._permutation._labels[0][side]
        bot = self._permutation._labels[1][side]

        unrank = self._permutation.alphabet().unrank
        top_letter = unrank(top)
        bot_letter = unrank(bot)

        length_top = self._lengths[top]
        length_bot = self._lengths[bot]

        if length_top > length_bot:
            winner = 0 # TODO: this value is ignored
            winner_interval = top
            loser_interval = bot
            abc = (bot_letter, bot_letter, top_letter)
            winner = 't'
        elif length_top < length_bot:
            winner = 1 # TODO: this value is ignored
            winner_interval = bot
            loser_interval = top
            abc = (top_letter, bot_letter, top_letter)
            winner = 'b'
        elif error_on_saddles:
            raise ValueError("saddle connection found")
        else:
            # we do a pseudo-top Rauzy induction and remove the bot interval
            # (the iet get one interval less)
            p = self._permutation
            top = p._labels[0][side]
            bot = p._labels[1][side]
            p._identify_intervals(side)
            self._lengths[top] = 0
            return None, (bot_letter,bot_letter,top_letter)

        self._permutation = self._permutation.rauzy_move(winner=winner, side=side, inplace=True)
        self._lengths[winner_interval] -= self._lengths[loser_interval]

        return winner, abc

    def _backward_rauzy_move(self, winner, side=-1):
        r"""
        Inplace backward Rauzy move.

        EXAMPLES::

            sage: from surface_dynamics import iet
            sage: x = polygen(QQ)
            sage: K.<cbrt2> = NumberField(x^3 - 2, embedding=AA(2)**(1/3))
            sage: p = iet.Permutation("a b c d", "d c b a")
            sage: T0 = iet.IntervalExchangeTransformation(p, [1, cbrt2, cbrt2**2, cbrt2 - 1])
            sage: T1 = copy(T0)
            sage: T1.lengths()
            (1, cbrt2, cbrt2^2, cbrt2 - 1)
            sage: T1._backward_rauzy_move(0)
            sage: T1.lengths()
            (1, cbrt2, cbrt2^2, cbrt2^2 + cbrt2 - 1)
            sage: _ = T1._rauzy_move()
            sage: T1 == T0
            True

            sage: T1._backward_rauzy_move(0)
            sage: T1._backward_rauzy_move(1)
            sage: T1._backward_rauzy_move(1)
            sage: T1._backward_rauzy_move(0)
            sage: _ = T1._rauzy_move()
            sage: _ = T1._rauzy_move()
            sage: _ = T1._rauzy_move()
            sage: _ = T1._rauzy_move()
            sage: T1 == T0
            True
        """
        self._permutation = self._permutation.backward_rauzy_move(winner=winner, side=side, inplace=True)
        winner_interval = self._permutation._labels[winner][side]
        loser_interval = self._permutation._labels[1-winner][side]
        self._lengths[winner_interval] += self._lengths[loser_interval]

    def _zorich_move(self, side=-1):
        r"""
        Performs a Zorich move (acceleration of Rauzy) inplace

        INPUT:

        - ``side`` - must be 0 or -1 (no verification)

        OUTPUT:

        - ``winner_letter`` - whether top or bot was done

        - ``m`` -- number of Rauzy made on the same side

        TESTS::

            sage: from surface_dynamics import *

            sage: p = iet.Permutation('a b', 'b a')
            sage: K.<sqrt3> = QuadraticField(3)
            sage: t = iet.IntervalExchangeTransformation(p, [1, sqrt3 - 1])
            sage: t._zorich_move()
            ('b', 1)
            sage: t._zorich_move()
            ('t', 2)
            sage: t._zorich_move()
            ('b', 1)
            sage: t._zorich_move()
            ('t', 2)
            sage: t._zorich_move()
            ('b', 1)
            sage: continued_fraction(sqrt3 - 1)
            [0; (1, 2)*]

            sage: x = polygen(ZZ)
            sage: K.<a> = NumberField(x^3 - 2, embedding=AA(2)**(1/3))
            sage: t = iet.IntervalExchangeTransformation(p, [1, a])
            sage: [t._zorich_move()[1] for _ in range(10)]
            [1, 3, 1, 5, 1, 1, 4, 1, 1, 8]
            sage: continued_fraction(a)
            [1; 3, 1, 5, 1, 1, 4, 1, 1, 8, 1, 14, 1, 10, 2, 1, 4, 12, 2, 3, ...]

        A self-similar example in genus 2::

            sage: x = polygen(ZZ)
            sage: poly = x^4 - 11*x^3 + 21*x^2 - 11*x + 1
            sage: l = max(poly.roots(AA, False))
            sage: K.<a> = NumberField(poly, embedding=l)
            sage: lengths = vector([-3*a^3 + 33*a^2 - 63*a + 33, -4*a^3 + 42*a^2 - 63*a + 14,
            ....:     7*a^3 - 74*a^2 + 116*a - 34, -a^2 + 10*a - 10])
            sage: p = iet.Permutation('a b c d', 'd c b a')
            sage: t = iet.IntervalExchangeTransformation(p, lengths)
            sage: [t._zorich_move() for _ in range(6)]
            [('t', 1), ('b', 2),  ('t', 2), ('b', 1), ('t', 2), ('b', 2)]
            sage: lengths == a * t.lengths()
            True
            sage: [t._zorich_move(side=0) for _ in range(6)]
            [('b', 1), ('t', 2), ('b', 2), ('t', 1), ('b', 2), ('t', 2)]
            sage: lengths == a^2 * t.lengths()
            True

        Saddle connection detection::

            sage: p = iet.Permutation('a b c', 'c b a')
            sage: T = iet.IntervalExchangeTransformation(p, [41, 1, 5])
            sage: T._zorich_move()
            Traceback (most recent call last):
            ...
            ValueError: saddle connection found
        """
        top = self._permutation._labels[0][side]
        bot = self._permutation._labels[1][side]

        length_top = self._lengths[top]
        length_bot = self._lengths[bot]

        if length_top > length_bot:
            winner = 0
            winner_letter = 't'
            winner_interval = top
            loser_interval = bot
        elif length_top < length_bot:
            winner = 1
            winner_letter = 'b'
            winner_interval = bot
            loser_interval = top
        else:
            raise ValueError("saddle connection found")

        # number of full loops
        lwin = self._lengths[winner_interval]
        loser = 1 - winner
        loser_row = self._permutation._labels[loser]
        llos = 0
        k = 0
        if side == -1:
            # right induction
            j = -1
            while loser_row[j] != winner_interval:
                llos += self._lengths[loser_row[j]]
                j -= 1
                k += 1
        else:
            # left induction
            j = 0
            while loser_row[j] != winner_interval:
                llos += self._lengths[loser_row[j]]
                j += 1
                k += 1

        # remove the full loops
        m = (lwin / llos).floor()
        self._lengths[winner_interval] -= m*llos

        # remaining steps
        r = 0
        while self._lengths[winner_interval] >= self._lengths[loser_interval]:
            self._lengths[winner_interval] -= self._lengths[loser_interval]
            self._permutation.rauzy_move(winner=winner, side=side, inplace=True)
            loser_interval = self._permutation._labels[loser][side]
            r += 1

        if self._lengths[winner_interval].is_zero():
            raise ValueError("saddle connection found")

        return winner_letter, k*m + r

    def __copy__(self):
        r"""
        Returns a copy of this circle exchange transformation.

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: t = iet.CircleExchangeTransformation(('a b','b a'),[1,1], [sqrt(2), 1/4])
            sage: s = copy(t)
            sage: s == t
            True
            sage: s is t
            False
        """
        res = self.__class__()
        res._base_ring = self._base_ring
        res._vector_space = self._vector_space
        res._permutation = copy(self._permutation)
        res._lengths = copy(self._lengths)
        res._rotation = self._rotation
        return res

    def plot_function(self,**d):
        r"""
        Return a plot of the interval exchange transformation as a
        function. Transform a circle of perimeter x into an interval of length x.

        INPUT:

        - Any option that is accepted by line2d

        OUTPUT:

        2d plot -- a plot of the iet as a function

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: t = iet.IntervalExchangeTransformation(('a b c d','d a c b'),[1,1,1,1])
            sage: t.plot_function(rgbcolor=(0,1,0))    # not tested (problem with matplotlib font cache)
            Graphics object consisting of 4 graphics primitives
        """
        from sage.plot.plot import Graphics
        from sage.plot.line import line2d

        G = Graphics()
        l = self.singularities()
        t = self._permutation._twin

        # add marked singularities at 0 and self.length()
        #l[0].insert(0,)

        for i in range(len(self._permutation)-1):
            j = t[0][i]
            if j == len(self._permutation)-1:
                x = self.length() - l[1][j]
                G += line2d([(l[0][i],l[1][j]),(l[0][i]+x,self.length())], **d)
                G += line2d([(l[0][i]+x,0),(l[0][i+1],l[1][0])], **d)
            else:
                #print(i,j)
                G += line2d([(l[0][i],l[1][j]),(l[0][i+1],l[1][j+1])], **d)
        i = len(self._permutation)-1
        j = t[0][i]
        if j == len(self._permutation)-1:
            y = self.length() - l[0][i]
            G += line2d([(l[0][i],l[1][j]),(self.length(),l[1][j]+y)], **d)
            G += line2d([(0,l[1][j]+y),(l[0][0],l[1][0])], **d)
        else:
            y = self.length() - l[0][i]
            G += line2d([(l[0][i],l[1][j]),(self.length(),l[1][j]+y)], **d)
            G += line2d([(0,l[1][j]+y),(l[0][0],l[1][j+1])], **d)
        return G

    def plot_two_intervals(
        self,
        position=(0,0),
        vertical_alignment = 'center',
        horizontal_alignment = 'left',
        interval_height=0.1,
        labels_height=0.05,
        fontsize=14,
        labels=True,
        colors=None):
        r"""
        Returns a picture of the interval exchange transformation.

        INPUT:

        - ``position`` - a 2-uple of the position

        - ``horizontal_alignment`` - left (default), center or right

        - ``labels`` - boolean (default: True)

        - ``fontsize`` - the size of the label


        OUTPUT:

        2d plot -- a plot of the two intervals (domain and range)

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: t = iet.IntervalExchangeTransformation(('a b','b a'),[1,1])
            sage: t.plot_two_intervals() # random output due to matplotlib deprecation warnings with SageMath <9.1
            Graphics object consisting of 8 graphics primitives
        """
        from sage.plot.plot import Graphics
        from sage.plot.line import line2d
        try:
            from sage.plot.plot import text
        except ImportError:
            from sage.plot.text import text
        from sage.plot.colors import rainbow

        G = Graphics()

        lengths = list(map(float, self._lengths))
        total_length = sum(lengths)

        if colors is None:
            colors = rainbow(len(self._permutation), 'rgbtuple')

        if horizontal_alignment == 'left':
            s = position[0]
        elif horizontal_alignment == 'center':
            s = position[0] - total_length / 2
        elif horizontal_alignment == 'right':
            s = position[0] - total_length
        else:
            raise ValueError("horizontal_alignement must be left, center or right")
        
        s += self._rotation[0]

        top_height = position[1] + interval_height
        for i in self._permutation._labels[0][:-1]:
#            if s+lengths[i] <= self.length():
            G += line2d([(s,top_height),(s+lengths[i],top_height)],
                rgbcolor=colors[i])
            if labels is True:
                G += text(str(self._permutation._alphabet.unrank(i)),
                    (s+float(lengths[i])/2,top_height+labels_height),
                    horizontal_alignment='center',
                    rgbcolor=colors[i],
                    fontsize=fontsize)

            s += lengths[i]

        i = self._permutation._labels[0][-1]
        G += line2d([(s,top_height),(self.length(),top_height)],
                rgbcolor=colors[i])
        G += line2d([(0,top_height),(s+lengths[i]-self.length(),top_height)],
                rgbcolor=colors[i])
        if labels is True:
            G += text(str(self._permutation._alphabet.unrank(i)),
                ((s+float(self.length()))/2,top_height+labels_height),
                horizontal_alignment='center',
                rgbcolor=colors[i],
                fontsize=fontsize)
            G += text(str(self._permutation._alphabet.unrank(i)),
                (float(s+lengths[i]-self.length())/2,top_height+labels_height),
                horizontal_alignment='center',
                rgbcolor=colors[i],
                fontsize=fontsize)

        if horizontal_alignment == 'left':
            s = position[0]
        elif horizontal_alignment == 'center':
            s = position[0] - total_length / 2
        elif horizontal_alignment == 'right':
            s = position[0] - total_length
        else:
            raise ValueError("horizontal_alignement must be left, center or right")

        s += self._rotation[1]

        bottom_height = position[1] - interval_height
        for i in self._permutation._labels[1][:-1]:
            G += line2d([(s,bottom_height), (s+lengths[i],bottom_height)],
                rgbcolor=colors[i])
            if labels is True:
                G += text(str(self._permutation._alphabet.unrank(i)),
                    (s+float(lengths[i])/2,bottom_height-labels_height),
                    horizontal_alignment='center',
                    rgbcolor=colors[i],
                    fontsize=fontsize)
            s += lengths[i]

        i = self._permutation._labels[1][-1]
        G += line2d([(s,bottom_height),(self.length(),bottom_height)],
                rgbcolor=colors[i])
        G += line2d([(0,bottom_height),(s+lengths[i]-self.length(),bottom_height)],
                rgbcolor=colors[i])
        if labels is True:
            G += text(str(self._permutation._alphabet.unrank(i)),
                ((s+float(self.length()))/2,bottom_height-labels_height),
                horizontal_alignment='center',
                rgbcolor=colors[i],
                fontsize=fontsize)
            G += text(str(self._permutation._alphabet.unrank(i)),
                (float(s+lengths[i]-self.length())/2,bottom_height-labels_height),
                horizontal_alignment='center',
                rgbcolor=colors[i],
                fontsize=fontsize)

        return G

    def plot_towers(self, iterations, position=(0,0), colors=None):
        """
        Plot the towers of this interval exchange obtained from Rauzy induction.

        INPUT:

        - ``nb_iterations`` -- the number of steps of Rauzy induction

        - ``colors`` -- (optional) colors for the towers

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: p = iet.Permutation('A B', 'B A')
            sage: T = iet.IntervalExchangeTransformation(p, [0.41510826, 0.58489174])
            sage: T.plot_towers(iterations=5)   # not tested (problem with matplotlib font cache)
            Graphics object consisting of 65 graphics primitives
        """
        px, py = map(float, position)

        T,_,towers = self.rauzy_move(iterations=iterations,data=True)
        pi = T.permutation()
        A = pi.alphabet()
        lengths = [float(length) for length in T.lengths()]

        if colors is None:
            from sage.plot.colors import rainbow
            colors = {a:z for a,z in zip(A, rainbow(len(A)))}

        from sage.plot.graphics import Graphics
        from sage.plot.line import line2d
        from sage.plot.polygon import polygon2d
        from sage.plot.text import text
        G = Graphics()
        x = px
        for letter in pi[0]:
            y = x + lengths[A.rank(letter)]
            tower = towers.image(letter)
            h = tower.length()
            G += line2d([(x,py),(x,py+h)], color='black')
            G += line2d([(y,py),(y,py+h)], color='black')
            for i,a in enumerate(tower):
                G += line2d([(x,py+i),(y,py+i)], color='black')
                G += polygon2d([(x,py+i),(y,py+i),(y,py+i+1),(x,py+i+1)], color=colors[a], alpha=0.4)
                G += text(a, ((x+y)/2, py+i+.5), color='darkgray')
            G += line2d([(x,py+h),(y,py+h)], color='black', linestyle='dashed')
            G += text(letter, ((x+y)/2, py+h+.5), color='black', fontsize='large')
            x = y
        x = px
        G += line2d([(px,py-.5),(px+sum(lengths),py-.5)], color='black')
        for letter in pi[1]:
            y = x + lengths[A.rank(letter)]
            G += line2d([(x,py-.7),(x,py-.3)], color='black')
            G += text(letter, ((x+y)/2, py-.5), color='black', fontsize='large')
            x = y
        G += line2d([(x,py-.7),(x,py-.3)], color='black')

        return G

    plot = plot_two_intervals

    def show(self):
        r"""
        Shows a picture of the interval exchange transformation

        EXAMPLES::

            sage: from surface_dynamics import *

            sage: phi = QQbar((sqrt(5)-1)/2)
            sage: t = iet.IntervalExchangeTransformation(('a b','b a'),[1,phi])
            sage: t.show() # not tested (problem with matplotlib font cache)
        """
        self.plot_two_intervals().show(axes=False)
