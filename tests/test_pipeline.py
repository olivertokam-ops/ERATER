import unittest
import os
import json

class TestEraterPipeline(unittest.TestCase):

    def test_config_file_exists(self):
        """Verifica che il file di configurazione config.json esista."""
        self.assertTrue(os.path.exists("config.json"), "Il file config.json non è stato trovato.")

    def test_config_structure(self):
        """Verifica che la configurazione contenga la soglia di prossimità."""
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)
            self.assertIn("proximity_threshold", config, "Manca la soglia di prossimità nel file config.json")

if __name__ == "__main__":
    unittest.main()
