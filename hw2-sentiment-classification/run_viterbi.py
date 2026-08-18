import numpy as np

def viterbi(initial_probs, transition_probs, emission_probs, observations, states):
    n_obs = len(observations)
    n_states = len(states)
    
    # Initialize delta and psi matrices
    delta = np.zeros((n_obs, n_states)) # store max prob
    psi = np.zeros((n_obs, n_states), dtype=int) # store state

    # Initialization (t=0)
    word = observations[0]
    for s_idx, state in enumerate(states):
        delta[0, s_idx] = initial_probs[state] * emission_probs[state].get(word, 0)

    # Recursion (t=1 to n_obs-1)
    for t in range(1, n_obs):
        word = observations[t]
        for s_idx, curr_state in enumerate(states):
            # Find max over previous states
            max_prob = -1
            max_state = 0

            for prev_s_idx, prev_state in enumerate(states):
                prob = (delta[t-1, prev_s_idx] * 
                       transition_probs[prev_state][curr_state] * 
                       emission_probs[curr_state].get(word, 0))
               
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_s_idx

            delta[t, s_idx] = max_prob
            psi[t, s_idx] = max_state
    print(delta)
    print(psi)
    # Termination
    best_final_state = np.argmax(delta[n_obs-1, :])

    # Backtracking
    path = [0] * n_obs
    path[n_obs-1] = best_final_state

    for t in range(n_obs-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    
    # Convert indices to state names and get probabilities
    most_likely_path = [states[idx] for idx in path]
    path_probs = [delta[t, path[t]] for t in range(n_obs)]
    
    return most_likely_path, path_probs


def main():
    states = ['N', 'M', 'V']
    
    initial_probs = {'N': 0.7, 'M': 0.1, 'V': 0.2}
    
    transition_probs = {
        'N': {'N': 0.2, 'M': 0.3, 'V': 0.5},
        'M': {'N': 0.4, 'M': 0.1, 'V': 0.5},
        'V': {'N': 0.8, 'M': 0.1, 'V': 0.1}
    }
    
    emission_probs = {
        'N': {'Patrick': 0.3, 'Cherry': 0.2, 'can': 0.1, 'will': 0.1, 'see': 0.1, 'spot': 0.2},
        'M': {'Patrick': 0, 'Cherry': 0, 'can': 0.4, 'will': 0.6, 'see': 0, 'spot': 0},
        'V': {'Patrick': 0, 'Cherry': 0, 'can': 0.1, 'will': 0.2, 'see': 0.5, 'spot': 0.2}
    }
    
    # c
    print("Part c: Patrick can see Cherry")
    obs_c = ['Patrick', 'can', 'see', 'Cherry']
    path_c, probs_c = viterbi(initial_probs, transition_probs, emission_probs, obs_c, states)
    
    for word, tag, prob in zip(obs_c, path_c, probs_c):
        print(f"  {word:10s} -> {tag} (probability: {prob:.10f})")
    print()
    
    # d
    print("Part d: will Cherry spot Patrick")
    obs_d = ['will', 'Cherry', 'spot', 'Patrick']
    path_d, probs_d = viterbi(initial_probs, transition_probs, emission_probs, obs_d, states)
    
    for word, tag, prob in zip(obs_d, path_d, probs_d):
        print(f"  {word:10s} -> {tag} (probability: {prob:.10f})")
    print()

if __name__ == "__main__":
    main()
