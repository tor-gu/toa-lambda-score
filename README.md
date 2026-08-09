# toa-lambda-score

This function implements the scoring algorithm for the Tournament of Albums.

The scoring system uses [MAP estimation](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) to fit the tournament results to a [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model).  See [below](#mathematical-details) for the mathematical details.

## Payload
```json
{
  "sd": 1.225,
  "unit_win_prob": 0.9,
  "results": [
    {"winner": "a", "loser": "b", "match_id": "1"},
    {"winner": "b", "loser": "c", "match_id": "2"},
    {"winner": "b", "loser": "d", "match_id": "3"},
    // ... more results 
  ]
}
```

| Field | Notes |
|---|---|
| `results` | required, non-empty array of dicts with `winner`, `loser`, and `match_id` ids. |
| `sd` | optional, overrides the `SD` env var. A number > 0.  |
| `unit_win_prob` | optional, overrides the `UNIT_WIN_PROB` env var. A number strictly between 0.5 and 1 |

See [parameters](#parameters) for the meaning of `sd` and `unit_win_prob`.

## Response
`{"statusCode": ..., "body": ...}` , and `body` is a JSON string encoding the scoring results:

```json
{
  "scores": [
    {"id": "a", "score": 0.478, "robustness": 1.83},
    {"id": "b", "score": -0.231, "robustness": 1.92},
    {"id": "c", "score": -1.432, "robustness": 3.34},
    // ... more scores
  ]
}
```

`score` and `robustness` are rounded to 3 decimals.

On invalid input the status is 400 and the body is `{"error": "..."}`. Anything else surfaces as a Lambda `FunctionError`.

## Parameters
Environment variables may be overridden by payload parameters.

| Env var | Payload field | Meaning |
|---|---|---|
| `SD` | `sd` | The standard deviation of the prior distribution of the MAP estimate. |
| `UNIT_WIN_PROB` | `unit_win_prob` | Scales the Bradley-Terry model. A value of 0.9 means that a score difference of 1.0 corresponds to a 90% probability of victory. |

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r dev/requirements-dev.txt
make check          # lint + tests
```

`make test` runs `tests/`, `make lint` runs black/isort/flake8, `make format` rewrites in place.

Run the scoring locally on a payload file:

```sh
./dev/run.sh dev/sample_payload.json
```

## Mathematical details
Players $P_1, P_2, \ldots$ have participated in a tournament with a series of head-to-head matchups, $M_1, M_2, \ldots$ . We are given the results of the matches, and want to estimate "scores" $x_1, x_2, ...$ that best explain the results of the tournament (and could be used to predict the results of additional matches).

The underlying assumptions are:

1. Scores are distributed normally about $0$:

      $$f(\mathbf{x};\sigma)=C \cdot exp \left(- { \frac{ \sum_{i=1}^n x_i^2}{2\sigma^2}} \right)$$

    This is the prior distribution in the MAP estimate.

2. Given two players, $P$ and $Q$ with scores $x$ and $y$, the probability that $P$ defeats $Q$ in a match is:

      $$B(x,y;\lambda) = \frac{e^{\lambda x}}{e^{\lambda x}+e^{\lambda y}}$$

   This is the Bradley-Terry model.

We assume that the scaling parameters $\sigma$ and $\lambda$ are given to us in advance.

### The objective function
Suppose there are $n$ players and $m$ matches, with winners $P_{i_1}, \ldots, P_{i_m}$ and losers $P_{j_1}, \ldots, P_{j_m}$.

We want to find the score values $\mathbf{x}=x_1,...,x_n$ that maximize the regularized likelihood function

$$L(\mathbf{x}) = f(\mathbf{x}) \cdot \prod_{k=1}^{m} B(x_{i_k},x_{j_k})$$

The log-likelihood then is

$$\log L(\mathbf{x}) = -\frac{1}{2\sigma^2} \sum_{i=1}^n x_i^2 + \lambda \sum_{k=1}^m x_{i_k} - \sum_{k=1}^m\log(e^{\lambda x_{i_k}} + e^{\lambda x_{j_k}}) + D$$

To maximize $L$, we can ignore $D$ and combine terms in $\log L$ to obtain an objective function

$$of(\mathbf{x};\sigma,\lambda) = -\frac{1}{2\sigma^2} \sum_{i=1}^n x_i^2 + \lambda \sum_{i=1}^nw_i x_i - \sum_{1\leq i \lt j \leq n} m_{i,j} \log(e^{\lambda x_i}+e^{\lambda x_j})$$

where $w_i$ is the number of wins by $P_i$ and $m_{i,j}$ is the number of matches between $P_i$ and $P_j$.

The value of $\hat{\mathbf{x}} = \hat{x}_1,\ldots,\hat{x}_n$ that maximizes the objective function $of(\mathbf{x})$ is our scoring estimate.

### The scaling parameters
There are two scaling parameters in the objective function, $of(\mathbf{x};\sigma,\lambda)$:

$\sigma$ is the standard deviation of the normal distribution, our prior distribution for the MAP estimate. It is passed to us directly.

$\lambda$ is the scaling parameter for the Bradley-Terry model. This is computed from $p$, the _unit win probability_. Given a match between two players whose score differ by exactly 1, we require the chance that the stronger player wins is $p$. We can compute $\lambda$ from $p$ by

$$\lambda = \log \left(\frac{p}{1-p}\right)$$

### Robustness
We also compute the _robustness_ of each player's score $\hat{x}_i$:

$$r_i = 4 \sum_k B(\hat{x}_i,\hat{x}_{o_k})\cdot B(\hat{x}_{o_k},\hat{x}_i)$$

where the sum is taken over all matches that $P_i$ participated in, and $P_{o_k}$ is the opponent $P_i$ faced in match $M_k$.

The robustness is related to the second partial derivative of the log-likelihood function:

$$\frac{\partial^2 \log(L)}{\partial x_i^2} =  -\frac{1}{\sigma^2} - \frac{\lambda^2 r_i}{4}$$

We choose the scaling factor of $4$ so that a "perfect match" (that is, a match between players whose score estimates are equal) contributes exactly $1$ to the robustness score of each player.

### Linear constraint
We use the [scipy optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) package to maximize the objective function.

The actual maximum of the objective function will satisfy

$$\sum_{i=1}^n x_i = 0$$

However, the estimate of the maximum returned by the numerical package is not guaranteed to satisfy it, so we add it as a [constraint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html#scipy.optimize.LinearConstraint).


## Related projects

These live in their own repositories.

- [toa-data-pipeline](https://github.com/tor-gu/toa-data-pipeline) — its scores-updater invokes this function once per date being scored.
- [toa-lambda-layer-common](https://github.com/tor-gu/toa-lambda-layer-common) — shared layer providing logging and HTTP-envelope helpers.
- [toa-lambda-layer-scipy](https://github.com/tor-gu/toa-lambda-layer-scipy) — Layer containing the scipy package we use to maximize the objective function.
