from gensim.parsing import PorterStemmer
global_stemmer = PorterStemmer()

def _get_param_matrices(vocabulary, sentence_terms):
    """
    Returns
    =======
    1. Top 300(or lesser, if vocab is short) most frequent terms(list)
    2. co-occurence matrix wrt the most frequent terms(dict)
    3. Dict containing Pg of most-frequent terms(dict)
    4. nw(no of terms affected) of each term(dict)
    """
 
    #Figure out top n terms with respect to mere occurences
    n = min(300, len(vocabulary))
    topterms = list(vocabulary.keys())
    topterms.sort(key = lambda x: vocabulary[x], reverse = True)
    topterms = topterms[:n]
 
    #nw maps term to the number of terms it 'affects'
    #(sum of number of terms in all sentences it
    #appears in)
    nw = {}
    #Co-occurence values are wrt top terms only
    co_occur = {}
    #Initially, co-occurence matrix is empty
    for x in vocabulary:
        co_occur[x] = [0 for i in range(len(topterms))]
 
    #Iterate over list of all sentences' vocabulary dictionaries
    #Build the co-occurence matrix
    for sentence in sentence_terms:
        total_terms = sum(list(sentence.values()))
        #This list contains the indices of all terms from topterms,
        #that are present in this sentence
        top_indices = []
        #Populate top_indices
        top_indices = [topterms.index(x) for x in sentence
                       if x in topterms]
        #Update nw dict, and co-occurence matrix
        for term in sentence:
            nw[term] = nw.get(term, 0) + total_terms
            for index in top_indices:
                co_occur[term][index] += (sentence[term] *
                                          sentence[topterms[index]])
 
    #Pg is just nw[term]/total vocabulary of text
    Pg = {}
    N = sum(list(vocabulary.values()))
    for x in topterms:
        Pg[x] = float(nw[x])/N
 
    return topterms, co_occur, Pg, nw
 
 
def get_top_n_terms(vocabulary, sentence_terms, n=50):
    """
    Returns the top 'n' terms from a block of text, in the form of a list,
    from most important to least.
 
    'vocabulary' should be a dict mapping each term to the number
    of its occurences in the entire text.
    'sentence_terms' should be an iterable of dicts, each denoting the
    vocabulary of the corresponding sentence.
    """
 
    #First compute the matrices
    topterms, co_occur, Pg, nw = _get_param_matrices(vocabulary,
                                                     sentence_terms)
 
    #This dict will map each term to its weightage with respect to the
    #document
    result = {}
 
    N = sum(list(vocabulary.values()))
    #Iterates over all terms in vocabulary
    for term in co_occur:
        term = str(term)
        org_term = str(term)
        for x in Pg:
            #expected_cooccur is the expected cooccurence of term with this
            #term, based on nw value of this and Pg value of the other
            expected_cooccur = nw[term] * Pg[x]
            #Result measures the difference(in no of terms) of expected
            #cooccurence and  actual cooccurence
            result[org_term] = ((co_occur[term][topterms.index(x)] -
                                 expected_cooccur)**2/ float(expected_cooccur))
 
    terms = list(result.keys())
    terms.sort(key=lambda x: result[x],
               reverse=True)
 
    return terms[:n]

class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """
 
    #This reverse lookup will remember the original forms of the stemmed
    #words
    word_lookup = {}
 
    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """
 
        #Stem the word
        stemmed = global_stemmer.stem(word)
 
        #Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)
 
        return stemmed
 
    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """
 
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word

from scipy.spatial.distance import cosine
from networkx import Graph
 
def build_mind_map(model, stemmer, root, nodes, alpha=0.2):
    """
    Returns the Mind-Map in the form of a NetworkX Graph instance.
 
    'model' should be an instance of gensim.models.Word2Vec
    'nodes' should be a list of terms, included in the vocabulary of
    'model'.
    'root' should be the node that is to be used as the root of the Mind
    Map graph.
    'stemmer' should be an instance of StemmingHelper.
    """
 
    #This will be the Mind-Map
    g = Graph()
 
    #Ensure that the every node is in the vocabulary of the Word2Vec
    #model, and that the root itself is included in the given nodes
    for node in nodes:
        if node not in model.wv.vocab:
            raise ValueError(node + " not in model's vocabulary")
    if root not in nodes:
        raise ValueError("root not in nodes")
 
    ##Containers for algorithm run
    #Initially, all nodes are unvisited
    unvisited_nodes = set(nodes)
    #Initially, no nodes are visited
    visited_nodes = set([])
    #The following will map visited node to its contextual vector
    visited_node_vectors = {}
    #Thw following will map unvisited nodes to (closest_distance, parent)
    #parent will obviously be a visited node
    node_distances = {}
 
    #Initialization with respect to root
    current_node = root
    visited_node_vectors[root] = model[root]
    unvisited_nodes.remove(root)
    visited_nodes.add(root)
 
    #Build the Mind-Map in n-1 iterations
    for i in range(1, len(nodes)):
        #For every unvisited node 'x'
        for x in unvisited_nodes:
            #Compute contextual distance between current node and x
            dist_from_current = cosine(visited_node_vectors[current_node],
                                       model[x])
            #Get the least contextual distance to x found until now
            distance = node_distances.get(x, (100, ''))
            #If current node provides a shorter path to x, update x's
            #distance and parent information
            if distance[0] > dist_from_current:
                node_distances[x] = (dist_from_current, current_node)
 
        #Choose next 'current' as that unvisited node, which has the
        #lowest contextual distance from any of the visited nodes
        next_node = min(unvisited_nodes,
                        key=lambda x: node_distances[x][0])
 
        ##Update all containers
        parent = node_distances[next_node][1]
        del node_distances[next_node]
        next_node_vect = ((1 - alpha)*model[next_node] +
                          alpha*visited_node_vectors[parent])
        visited_node_vectors[next_node] = next_node_vect
        unvisited_nodes.remove(next_node)
        visited_nodes.add(next_node)
 
        #Add the link between newly selected node and its parent(from the
        #visited nodes) to the NetworkX Graph instance
        g.add_edge(stemmer.original_form(parent).capitalize(),
                   stemmer.original_form(next_node).capitalize())
 
        #The new node becomes the current node for the next iteration
        current_node = next_node
 
    return g
