Combinadics
===========

A fast combinations calculation in jax.

Idea of combinadic implementation is from
https://jamesmccaffrey.wordpress.com/2022/06/28/generating-the-mth-lexicographical-element-of-a-combination-using-the-combinadic
and some useful information can be found here: https://en.wikipedia.org/wiki/Combinatorial_number_system. Below I copied and aggregated some of the details.

## Introduction
The following code demostrates the combinations calculation in numpy and via combinadics:
```python
    # setup
    n = 4
    k = 3
    totalcount = math.comb(n, k)

    # numpy
    print(f"Calculate combinations \"{n} choose {k}\" in numpy:")
    for comb in itertools.combinations(np.arange(start=0, stop=n, dtype=jnp.int32), k):
        print(comb)

    # combinadics
    print("Calculate via combinadics:")
    actual = n-1 - calculateMth(n, k, totalcount-1 - jnp.arange(start=0, stop=n, dtype=jnp.int32),)
    for comb in actual:
        print(comb)
```
And the output from execution of the code is:
```
Calculate combinations "4 choose 3" in numpy:
(0, 1, 2)
(0, 1, 3)
(0, 2, 3)
(1, 2, 3)
Calculate via combinadics:
[0 1 2]
[0 1 3]
[0 2 3]
[1 2 3]
```

## A bit of theory

You can think of a combinadic as an alternate representation of an integer. Consider the integer $859$. It can be represented as the sum of powers of $10$ as

$$
859 = 8 \times 10^2 + 5 \times 10^1 + 9 \times 10^0
$$

<!-- ``` -->
<!-- 859 = (8 * 10^2) + (5 * 10^1) + (9 * 10^0) -->
<!-- ``` -->
The combinadic of an integer is its representation based on a variable base corresponding to the values of the binomial coefficient $\dbinom{n}{k}$. For example if ($n=7, k=4$) then the integer $27$ can be represented as
$$
27 = \dbinom{{6}}{4} + \dbinom{5}{3} + \dbinom{2}{2} + \dbinom{1}{1} = 15 + 10 + 1 + 1
$$

<!-- ``` -->
<!-- 27 = Choose(6,4) + Choose(5,3) + Choose(2,2) + Choose(1,1) = 15 + 10 + 1 + 1 -->
<!-- ``` -->
With ($n=7, k=4$), any number $m$ between $0$ and $34$ (the total number of combination elements for $n$ and $k$) can be uniquely represented as
$$
m = \dbinom{c_1}{4} + \dbinom{c_2}{3}+\dbinom{c_3}{2} + \dbinom{c_4}{1}
$$

<!-- ``` -->
<!-- m = Choose(c1,4) + Choose(c2,3) + Choose(c3,2) + Choose(c4,1) -->
<!-- ``` -->
where $n > c_1 > c_2 > c_3 > c_4$. Notice that $n$ is analogous to the base because all combinadic digits are between $0$ and $n-1$ (just like all digits in ordinary base $10$ are between $0$ and $9$). The value of $k$ determines the number of terms in the combinadic.

---
Here’s an example of how a combinadic is calculated. Suppose you are working with ($n=7, k=4$) combinations, and $m = 8$. You want the combinadic of 8 because, as it turns out, the combinadic can be converted to combination element [8].

The combinadic of 8 will have the form:
$$
8 = \dbinom{c_1}{4} + \dbinom{c_2}{3}+\dbinom{c_3}{2} + \dbinom{c_4}{1}
$$
The first step is to determine the value of c1. We try c1 = 6 (the largest number less than n = 7) and get Choose(6,4) = 15, which is too large because we’re over 8. Next, we try c1 = 5 and get Choose(5,4) = 5, which is less than 8, so bingo, c1 = 5.

At this point we have used up 5 of the original number m=8 so we have 3 left to account for. To determine the value of c2, we try 4 (the largest number less than the 5 we got for c1), but get Choose(4,3) = 4, which is barely too large. Working down we get to c2 = 3 and Choose(3,3) = 1, so c2 = 3.

We used up 1 of the remaining 3 we had to account for, so we have 2 left to consume. Using the same ideas we’ll get c3 = 2 with Choose(2,2) = 1, so we have 1 left to account for. And then we’ll find that c4 = 1 because Choose(1,1) = 1. Putting our four c values together we conclude that the combinadic of m=8 for (n=7, k=4) combinations is ( 5 3 2 1 ).  

---

Suppose (n=7, k=4). There are Choose(7,4) = 35 combination elements, indexed from 0 to 34. The **dual** lexicographic indexes are the ones on opposite ends so to speak: indexes 0 and 34 are duals, indexes 1 and 33 are duals, indexes 2 and 32 are duals, and so forth. Notice that each pair of dual indexes sum to 34, so if you know any index it is easy to find its dual.

Now, continuing the first example above for the number m=27 with (n=7, k=4), suppose you are able to find the combinadic of 27 and get ( 6 5 2 1 ). Now suppose you subtract each digit in the combinadic from n-1 = 6 and get ( 0 1 4 5 ). Amazingly, this gives you the combination element [7], the dual index of 27. Putting these ideas together you have an elegant algorithm to determine an arbitrarily specified combination element for given n and k values. To find the combination element for index m, first find its dual and call it x. Next, find the combinadic of x. Then subtract each digit of the combinadic of x from n-1 and the result is the mth lexicographic combination element.

The table below shows the relationships among m, the dual of m, combination element [m], the combinadic of m, and (n-1) – ci for (n=5, k=3).
```
m dual(m) Element(m) combinadic(m) (n-1) - ci
==============================================
[0]  9    { 0 1 2 }   ( 2 1 0 )     ( 2 3 4 )
[1]  8    { 0 1 3 }   ( 3 1 0 )     ( 1 3 4 )
[2]  7    { 0 1 4 }   ( 3 2 0 )     ( 1 2 4 )
[3]  6    { 0 2 3 }   ( 3 2 1 )     ( 1 2 3 )
[4]  5    { 0 2 4 }   ( 4 1 0 )     ( 0 3 4 )
[5]  4    { 0 3 4 }   ( 4 2 0 )     ( 0 2 4 )
[6]  3    { 1 2 3 }   ( 4 2 1 )     ( 0 2 3 )
[7]  2    { 1 2 4 }   ( 4 3 0 )     ( 0 1 4 )
[8]  1    { 1 3 4 }   ( 4 3 1 )     ( 0 1 3 )
[9]  0    { 2 3 4 }   ( 4 3 2 )     ( 0 1 2 )
```

## Limitations

64-bit numbers  
Performance of a single GPU
