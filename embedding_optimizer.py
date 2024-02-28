import torch
import torch.nn.functional as F

class EmbeddingOptimizer:
    """
    Class for optimizing image embeddings using Projected Gradient Descent.

    Attributes:
        model (torch.nn.Module): Neural network model for generating embeddings.
        learning_rate (float): Learning rate for gradient descent.
        epsilon (float): Epsilon value for controlling perturbations in PGD.
    """

    def __init__(self, model, learning_rate, epsilon):
        """
        Initializes the EmbeddingOptimizer with a model, learning rate, and epsilon.
        Args:
            model (torch.nn.Module): Model for generating embeddings.
            learning_rate (float): Learning rate for gradient descent.
            epsilon (float): Epsilon value for controlling perturbations in PGD.
        """
        self.model = model
        self.lr = learning_rate
        self.epsilon = epsilon

    def optimize_embeddings(self, cur_input, target_emb, l2_dist_threshold, cosine_sim_threshold):
        """
        Adjusts initial input to match target embedding, stopping when thresholds are met.
        Args:
            cur_input (torch.Tensor): Input tensor to be optimized.
            target_emb (torch.Tensor): Target embedding to match.
            l2_dist_threshold (float): Threshold for squared L2 distance.
            cosine_sim_threshold (float): Threshold for cosine similarity.
        Returns:
            torch.Tensor: Optimized input tensor.
            list: L1 distances over iterations.
            list: Cosine similarities over iterations.
            list: Losses over iterations.
        """
        org_input = cur_input.clone()

        squared_l2_distance = float('inf')
        cosine_sim_arr = []
        loss_arr = []
        l1_dist_arr = []

        iteration_count = 0
        while squared_l2_distance >= l2_dist_threshold or cosine_sim <= cosine_sim_threshold:
            cur_input = cur_input.clone().detach().requires_grad_(True)

            cur_emb = self.model.encode_image(cur_input)

            loss = F.mse_loss(target_emb, cur_emb)
            loss_arr.append(loss.item())

            cur_input.grad = None
            loss.backward(retain_graph=True)
            grad = cur_input.grad

            updated_input = cur_input - self.lr * grad

            # Applied projection to ensure that updated_input is within the range
            projected_input = cur_input + torch.clamp(updated_input - cur_input, -self.epsilon, self.epsilon)

            with torch.no_grad():
                updated_emb = self.model.encode_image(projected_input)

            squared_l2_distance = torch.sum((target_emb - updated_emb)**2).item()

            updated_l1_dist = torch.sum(torch.abs(projected_input.detach() - org_input)).item()
            cosine_sim = F.cosine_similarity(target_emb, updated_emb)

            print(iteration_count, '\n')
            print("Squared L2 Distance:", squared_l2_distance)
            print("Cosine Similarity:", cosine_sim.detach().cpu().item())

            l1_dist_arr.append(updated_l1_dist)
            cosine_sim_arr.append(cosine_sim.detach().cpu().item())

            cur_input = projected_input
            iteration_count += 1

        return projected_input.detach(), l1_dist_arr, cosine_sim_arr, loss_arr

