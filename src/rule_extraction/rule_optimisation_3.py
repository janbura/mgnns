# # Dictionary mapping each variables in gamma_i to its relevant positions (given by label of layer 0).
# total_variable_to_positions_dict = {}
# for (y, j) in contributors_to_influence_dict:
#     if y not in total_variable_to_positions_dict:
#         total_variable_to_positions_dict[y] = {j}
#     else:
#         total_variable_to_positions_dict[y].add(j)
#
# def get_successors(partial_variable_to_positions_dict, conjunction_form):
#     successors = []
#     for y in total_variable_to_positions_dict:
#         if y not in partial_variable_to_positions_dict:
#             for j in total_variable_to_positions_dict[y]:
#                 new_successor = partial_variable_to_positions_dict.copy()
#                 new_successor[y] = {j}
#                 new_conjunction_form = conjunction_form.union(get_conjunction_for_contributor(y, j))
#                 successors.append((len(new_conjunction_form), new_successor, new_conjunction_form))
#         else:
#             for j in total_variable_to_positions_dict[y]:
#                 if j not in partial_variable_to_positions_dict[y]:
#                     new_successor = partial_variable_to_positions_dict.copy()
#                     new_successor[y].add(j)
#                     new_conjunction_form = conjunction_form.union(
#                         get_conjunction_for_contributor(y, j))
#                     successors.append(
#                         (len(new_conjunction_form), new_successor, new_conjunction_form))
#     return successors

frontier = get_successors({}, set())
while frontier:
    # Sort in decreasing order of cost
    frontier = sorted(frontier, reverse=True)
    (score, dictionary, conjunction) = frontier.pop()
    (gr_features, node_to_gr_row_dict, gr_edge_list,
     gr_colour_list) = can_encoder_decoder.encode_dataset(
        conjunction)
    gr_dataset = Data(x=gr_features, edge_index=gr_edge_list, edge_type=gr_colour_list).to(device)
    gnn_output_gr = model(gr_dataset)
    if (gnn_output_gr[node_to_gr_row_dict[nodes.const_node_dict[x1]]][cd_pred_pos] >=
            cfg.derivation_threshold):
        frontier = None
        rule_body = conjunction
    else:
        frontier = list(set(frontier).union(set(get_successors(dictionary, conjunction))))

    rule_body = remove_redundant_atoms(rule_body)