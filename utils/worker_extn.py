import gc
import time
import torch

class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    """

    def save_self_initial_weights(self,):
        """Save a copy of itself in the CPU memory."""
        self.initial_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self.initial_weights[name] = p.detach().clone().cpu()
        print("Initial weights saved.")

    def restore_self_initial_weights(self,):
        """Restore the initial weights from CPU memory."""
        for name, p in self.model_runner.model.named_parameters():
            if name in self.initial_weights:
                p.data.copy_(self.initial_weights[name].to(p.device))

        del self.initial_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(1)  # wait for a while to ensure the weights are restored
        print("Initial weights restored.")

    def perturb_self_weights(self, seed, sigma_or_scale, coeff=1.0, negate=False):
        """
        Add noise(seed) scaled by sigma_or_scale * coeff (or subtract when negate=True).
        - For exploration:  perturb_self_weights(seed, SIGMA, 1.0, False)
          and restore with restore_self_weights(seed, SIGMA) as before.
        - For ES update:   perturb_self_weights(seed, 1.0, coeff, False)
          where coeff = ALPHA/POPULATION_SIZE * norm_reward.
        """
        scale = float(sigma_or_scale) * float(coeff)
        sign = -1.0 if negate else 1.0

        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"Weights {'restored' if negate else 'perturbed'} with scale={sign * scale}.")

    def restore_self_weights(self, seed, SIGMA):
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(seed)
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-SIGMA * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("Weights restored.")

    def save_self_weights_to_disk(self, filepath):
        """Save the current model weights to disk."""
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        print(f"Model weights saved to {filepath}.")