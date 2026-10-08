#include <iostream>
#include <vector>
#include <string>

using namespace std;

// function to count keyword matches
int scoreChunk(string chunk, string query){
  int score = 0;
  // initially we set the score as 0 and iterate throung the chunk and find keywords that matches as we instructed
  // If it matches, it gives a +1 incrimentation.
  // for now the keywords are AI and model

  if (chunk.find("AI") != string::npos)  score ++;
  if (chunk.find("model") != string::npos) score ++;
  return score;

}

int main(){
  // We provide a list of chunks and a query(for now that has no work)

  vector<string> chunks = {
    "This is an ai model which can perform many things",
    "Bannana is yellow in color",
    "model training is important"
  };
  string query = "ai model";
  // We then iterate through the chunks and then see if the function we made catches any keywords or not 
  for (int i = 0; i < chunks.size(); i++){
    int score = scoreChunk(chunks[i], query);
    cout << "chunk: " << chunks[i] << endl;
    cout << "score: " << score << endl << endl;
  }
  return 0;
}