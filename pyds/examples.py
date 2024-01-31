"""
Shows different use cases of the library.
"""

from itertools import product

from pyds import MassFunction

print("=== creating mass functions ===")
m1 = MassFunction({"ab": 0.6, "bc": 0.3, "a": 0.1, "ad": 0.0})  # using a dictionary
print("m_1 =", m1)
m2 = MassFunction([({"a", "b", "c"}, 0.2), ({"a", "c"}, 0.5), ({"c"}, 0.3)])  # using a list of tuples
print("m_2 =", m2)
m3 = MassFunction()
m3["bc"] = 0.8
m3[{}] = 0.2
print("m_3 =", m3, "(unnormalized mass function)")

print("\n=== belief, plausibility, and commonality ===")
print("bel_1({a, b}) =", m1.bel({"a", "b"}))
print("pl_1({a, b}) =", m1.pl({"a", "b"}))
print("q_1({a, b}) =", m1.q({"a", "b"}))
print("bel_1 =", m1.bel())  # entire belief function
print("bel_3 =", m3.bel())
print("m_3 from bel_3 =", MassFunction.from_bel(m3.bel()))  # construct a mass function from a belief function

print("\n=== frame of discernment, focal sets, and core  ===")
print("frame of discernment of m_1 =", m1.frame())
print("focal sets of m_1 =", m1.focal())
print("core of m_1 =", m1.core())
print("combined core of m_1 and m_3 =", m1.core(m3))

print("\n=== Dempster's combination rule, unnormalized conjunctive combination (exact and approximate) ===")
print("Dempster's combination rule for m_1 and m_2 =", m1 & m2)
print(
    "Dempster's combination rule for m_1 and m_2 (Monte-Carlo, importance sampling) =",
    m1.combine_conjunctive(m2, sample_count=1000, importance_sampling=True),
)
print("Dempster's combination rule for m_1, m_2, and m_3 =", m1.combine_conjunctive(m2, m3))
print("unnormalized conjunctive combination of m_1 and m_2 =", m1.combine_conjunctive(m2, normalization=False))
print(
    "unnormalized conjunctive combination of m_1 and m_2 (Monte-Carlo) =",
    m1.combine_conjunctive(m2, normalization=False, sample_count=1000),
)
print(
    "unnormalized conjunctive combination of m_1, m_2, and m_3 =", m1.combine_conjunctive([m2, m3], normalization=False)
)

print("\n=== normalized and unnormalized conditioning ===")
print("normalized conditioning of m_1 with {a, b} =", m1.condition({"a", "b"}))
print("unnormalized conditioning of m_1 with {b, c} =", m1.condition({"b", "c"}, normalization=False))

print("\n=== disjunctive combination rule (exact and approximate) ===")
print("disjunctive combination of m_1 and m_2 =", m1 | m2)
print("disjunctive combination of m_1 and m_2 (Monte-Carlo) =", m1.combine_disjunctive(m2, sample_count=1000))
print("disjunctive combination of m_1, m_2, and m_3 =", m1.combine_disjunctive([m2, m3]))

print("\n=== weight of conflict ===")
print("weight of conflict between m_1 and m_2 =", m1.conflict(m2))
print("weight of conflict between m_1 and m_2 (Monte-Carlo) =", m1.conflict(m2, sample_count=1000))
print("weight of conflict between m_1, m_2, and m_3 =", m1.conflict([m2, m3]))

print("\n=== pignistic transformation ===")
print("pignistic transformation of m_1 =", m1.pignistic())
print("pignistic transformation of m_2 =", m2.pignistic())
print("pignistic transformation of m_3 =", m3.pignistic())

print("\n=== local conflict uncertainty measure ===")
print("local conflict of m_1 =", m1.local_conflict())
print("entropy of the pignistic transformation of m_3 =", m3.pignistic().local_conflict())

print("\n=== sampling ===")
print("random samples drawn from m_1 =", m1.sample(5, quantization=False))
print("sample frequencies of m_1 =", m1.sample(1000, quantization=False, as_dict=True))
print("quantization of m_1 =", m1.sample(1000, as_dict=True))

print("\n=== map: vacuous extension and projection ===")
extended = m1.map(lambda h: product(h, {1, 2}))
print("vacuous extension of m_1 to {1, 2} =", extended)
projected = extended.map(lambda h: (t[0] for t in h))
print("project m_1 back to its original frame =", projected)

print("\n=== construct belief from data ===")
hist = {"a": 2, "b": 0, "c": 1}
print("histogram:", hist)
print("maximum likelihood:", MassFunction.from_samples(hist, "bayesian", s=0))
print("Laplace smoothing:", MassFunction.from_samples(hist, "bayesian", s=1))
print("IDM:", MassFunction.from_samples(hist, "idm"))
print("MaxBel:", MassFunction.from_samples(hist, "maxbel"))
print("MCD:", MassFunction.from_samples(hist, "mcd"))

print("\n=== The Murder of Mr. Jones ===")
print("There are three suspects to the case")
mk0 = MassFunction([({"Peter", "Paul", "Mary"}, 1)])
print(f"m_0 = {mk0}")

print("Evidence 1: the gender of the killer will be decided based on a dice roll")
mk1 = MassFunction([({"Peter", "Paul"}, 0.5), ({"Mary"}, 0.5)])
print(f"m_1 = {mk1}")

mk1 = mk1.combine_conjunctive(mk0)
print(f"m_01 = {mk1}")

print("Evidence 2: Peter has a perfect alibi => Peter is not the killer")
mk2 = MassFunction([({"Paul", "Mary"}, 1)])
print(f"m_2 = {mk2}")

mk1 = mk1.combine_conjunctive(mk2)
print(f"m_012 = {mk1}")
print(
    "The belief mass of {Peter, Paul} was transferred to {Paul}.\n"
    "Betting on the gender still produces the same odds."
)

print("\n=== The Unreliable Sensor Paradigm ===")
print(
    "Consider a sensor which checks the temperature.\n"
    "Temperature can be either hot(H) or cold(C).\n"
    "The sensor shines a light to indicate this, either red(R) for hot or blue(B) for cold.\n"
    "The sensor is unreliable, we will note this as working(W) and broken(B)."
)

print("The probability of being broken is 20%.")
ms = MassFunction([({"RHW", "BCW"}, 0.8), ({"RCB", "RHB", "BCB", "BHB"}, 0.2)])
print(f"m = {ms}")

print("After using the sensor, the light indicator is red(R), so how much do we believe the temperature to be Hot(H)?")
msr = MassFunction([({"RHW", "RCB", "RHB"}, 1)])
print(f"m_red = {msr}")

ms = ms.combine_conjunctive(msr)
print(f"m' = {ms}")
print("bel_m(Hot) =", ms.bel({"RHW", "RHB"}))
print("bel_m(Cold) =", ms.bel({"RCB"}))
print(f"pl_m(Hot) =", ms.pl({"RHW", "RHB"}))
print(f"pl_m(Cold) =", ms.pl({"RCB"}))
print(ms.pignistic())

print("\n=== The Zadeth's example ===")
print(
    "Lofti Zadeh describes an information fusion problem.\n"
    "A patient has an illness that can be caused by three different factors A, B or C.\n"
    "Doctor 1 says that the patient's illness is very likely to be caused by A\n"
    "(very likely, meaning probability p = 0.95), but B is also possible but not likely (p = 0.05)."
)
md1 = MassFunction([("A", 0.95), ("B", 0.05)])
print(f"m1 = {md1}")

print("Doctor 2 says that the cause is very likely C (p = 0.95), but B is also possible but not likely (p = 0.05).")
md2 = MassFunction([("C", 0.95), ("B", 0.05)])
print(f"m2 = {md2}")

print("How is one to make one's own opinion from this?")
print(f"Depster's rule of combination (DST default): {md1.combine_conjunctive(md2)}")
print(
    f"Depster's rule of combination, assuming open world ~ unnormalized (TBM default):"
    f" {md1.combine_conjunctive(md2, normalization=False)}"
)

md12 = md1.combine_disjunctive(md2)
print(f"Using the disjunctive rule: {md12}")
print(f"pignistic: {md12.pignistic()}")

print("\n=== Trying to model device categorization ===")
ms3 = MassFunction([("AB", 0.9), ("C", 0.1)])
ms1 = MassFunction(
    [
        ("A", 0.15),
        ("CB", 0.85),
    ]
)
ms2 = MassFunction([("A", 0.5), ("C", 0.5)])

ms13 = ms1.combine_disjunctive(ms3)
print(f"disjunctive join: {ms13}")

ms123 = ms13.combine_conjunctive(ms2)
print(f"conjunctive join: {ms123}")
print(f"pignistic: {ms123.pignistic()}")
