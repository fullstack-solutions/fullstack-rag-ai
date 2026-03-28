import requests
from typing import List, Dict
from .base import BaseSource

class GitHubSource(BaseSource):
    """
    Fetch GitHub repositories and their files.
    - Can operate fully in memory (no local clone)
    - Optionally clone to disk if clone_dir is specified
    """

    def __init__(self, token: str, clone_dir: str = None):
        """
        :param token: GitHub personal access token
        :param clone_dir: optional local directory to clone repos
        """
        self.token = token
        self.clone_dir = clone_dir

    def fetch_repos(self):
        """Fetch user repositories from GitHub API"""
        headers = {"Authorization": f"Bearer {self.token}"}
        page = 1
        response = []
        while True:
            url = f"https://api.github.com/user/repos?page={page}&per_page=100"
            res = requests.get(url, headers=headers)
            res.raise_for_status()
            data = res.json()
            response.extend(data)
            if not data:
                break
            for repo in data:
                yield repo
            page += 1
        return response

    def fetch_repo_files(self, owner: str, repo_name: str, branch="master") -> List[Dict]:
        """
        Fetch all files of a repository via GitHub API
        Returns list of dicts: {'path': ..., 'content': bytes}
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{branch}"
        res = requests.get(api_url, headers=headers)
        res.raise_for_status()
        tree = res.json().get("tree", [])

        files = []
        for node in tree:
            if node["type"] == "blob":
                file_url = node["url"]
                f_res = requests.get(file_url, headers=headers)
                f_res.raise_for_status()
                file_content = f_res.json().get("content", "")
                encoding = f_res.json().get("encoding", "base64")
                if encoding == "base64":
                    import base64
                    file_bytes = base64.b64decode(file_content)
                else:
                    file_bytes = file_content.encode("utf-8")

                files.append({
                    "path": node["path"],
                    "content": file_bytes
                })

        return files

    def load(self) -> List[Dict]:
        """
        Load repositories and their files.
        If clone_dir is specified, optionally clone to disk.
        Returns:
            List of repositories, each as dict:
            {
                "name": repo_name,
                "owner": owner,
                "files": [{"path": path, "content": bytes}]
            }
        """
        repos = self.fetch_repos()
        all_repos = []

        for repo in repos:
            try:
                owner = repo["owner"]["login"]
                name = repo["name"]
                files = self.fetch_repo_files(owner, name)
                repo_dict = {
                    "name": name,
                    "owner": owner,
                    "files": files
                }

                all_repos.append(repo_dict)

                # Optional: clone to disk if clone_dir is set
                if self.clone_dir:
                    import subprocess, os
                    os.makedirs(self.clone_dir, exist_ok=True)
                    repo_path = os.path.join(self.clone_dir, name)
                    if not os.path.exists(repo_path):
                        subprocess.run(
                            ["git", "clone", repo["clone_url"], repo_path],
                            check=True
                        )
                    else:
                        subprocess.run(
                            ["git", "-C", repo_path, "pull"],
                            check=True
                        )

            except Exception as e:
                print(f"[ERROR] Repo {repo['name']}: {e}")

        return all_repos