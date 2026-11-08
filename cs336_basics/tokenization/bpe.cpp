#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <chrono>
#include <cassert>

constexpr uint64_t N = 40'000;

constexpr uint64_t pack(uint32_t a, uint32_t b) {
    return (uint64_t(a) << 32) | b;
}

constexpr std::pair<uint32_t, uint32_t> unpack(uint64_t key) {
    return {uint32_t(key >> 32), uint32_t(key)};
}

struct Entry {
    uint64_t count;
    uint64_t key;
    uint64_t version;
};

std::vector<std::pair<std::string, std::string>> train_bpe(
    std::vector<std::vector<uint32_t>>& words,
    const std::vector<uint64_t>& freqs,
    uint32_t num_merges) {

    assert(words.size() == freqs.size());
    std::vector<std::string> vocab;
    vocab.reserve(256 + num_merges);

    for (size_t i = 0; i < 256; i++) {
        vocab.emplace_back(1, static_cast<char>(i));
    }

    assert(vocab.size() <= UINT32_MAX);

    auto cmp = [&](const Entry& x, const Entry& y) {
        if (x.count != y.count) return x.count < y.count;
        auto [a1, b1] = unpack(x.key);
        auto [a2, b2] = unpack(y.key);
        if (vocab[a1] != vocab[a2]) return vocab[a1] < vocab[a2];
        return vocab[b1] < vocab[b2];
    };

    std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> heap(cmp);

    std::vector<std::pair<std::string, std::string>> merges{};
    merges.reserve(num_merges);

    std::unordered_map<uint64_t, uint64_t> pair_counts;
    pair_counts.reserve(N);
    std::unordered_map<uint64_t, std::unordered_set<uint32_t>> pair_words;
    pair_words.reserve(N);
    std::unordered_map<uint64_t, uint64_t> versions;
    pair_words.reserve(N * 2);

    for (size_t i = 0; i < words.size(); ++i) {
        const auto& word = words[i];
        const uint64_t freq = freqs[i];

        for (size_t j = 0; j + 1 < word.size(); ++j) {
            uint64_t key = pack(word[j], word[j + 1]);

            pair_counts[key] += freq;
            pair_words[key].insert(i);
            versions[key] = 1;
        }
    }

    for (const auto& [key, count] : pair_counts) {
        heap.push(Entry{count, key, versions[key]});
    }

    for (size_t i = 0; i < num_merges; i++) {
        if (pair_counts.empty()) {
            break;
        }

        Entry best{};
        bool found = false;

        while (!heap.empty()) {
            Entry candidate = heap.top();
            heap.pop();

            auto it = pair_counts.find(candidate.key);

            if (it == pair_counts.end())
                continue;
            
            if (candidate.version != versions.at(candidate.key))
                continue;

            best = candidate;
            found = true;
            break;
        }

        if (!found) break;

        const uint64_t best_key = best.key;
        auto [a, b] = unpack(best_key);
        uint32_t new_id = vocab.size();
        merges.emplace_back(vocab[a], vocab[b]);
        vocab.push_back(vocab[a] + vocab[b]);

        assert(pair_words.contains(best_key));
        assert(!pair_words.at(best_key).empty());

        std::vector<uint32_t> ids(pair_words[best_key].begin(), pair_words[best_key].end());
        std::unordered_set<uint64_t> changed_pairs;

        for (uint32_t idx : ids) {
            const auto& old = words[idx];
            std::vector<uint32_t> new_word;
            size_t length = old.size();
            new_word.reserve(length);
            size_t k = 0;
            while (k < length) {
                if (k + 1 < length && old[k] == a && old[k + 1] == b) {
                    new_word.push_back(new_id);
                    k += 2;
                }
                else {
                    new_word.push_back(old[k]);
                    k += 1;
                }
            }

            if (new_word.size() == length) continue;

            uint64_t f = freqs[idx];
            for (size_t l = 0; l + 1 < length; l++) {
                uint32_t m = old[l];
                uint32_t n = old[l+1];
                uint64_t key = pack(m, n);

                auto it = pair_counts.find(key);
                if (it != pair_counts.end()) {

                    if (it->second <= f)
                        pair_counts.erase(it);
                    else
                        it->second -= f;
                    
                    changed_pairs.insert(key);
                }
            }

            for (size_t j = 0; j + 1 < new_word.size(); j++) {
                uint32_t x = new_word[j];
                uint32_t y = new_word[j+1];
                uint64_t key = pack(x, y);
                pair_counts[key] += f;
                changed_pairs.insert(key);
                pair_words[key].insert(idx);
            }

            std::unordered_set<uint64_t> old_pairs, new_pairs;

            for (size_t j = 0; j + 1 < old.size(); ++j)
                old_pairs.insert(pack(old[j], old[j + 1]));

            for (size_t j = 0; j + 1 < new_word.size(); ++j)
                new_pairs.insert(pack(new_word[j], new_word[j + 1]));
            
            for (uint64_t p : old_pairs) {
                if (new_pairs.contains(p)) continue; // -std=c++20

                auto it = pair_words.find(p);
                if (it == pair_words.end()) continue;

                it->second.erase(idx);

                if (it->second.empty())
                    pair_words.erase(it);
            }

            words[idx] = std::move(new_word);
        }

        for (uint64_t key : changed_pairs) {
            if (key == best_key) continue;
            uint64_t version = ++versions[key];

            auto it = pair_counts.find(key);
            if (it != pair_counts.end()) {
                heap.push(Entry{it->second, key, version});
            }
        }

        pair_words.erase(best_key);
        pair_counts.erase(best_key);

        if ((i + 1) % 100 == 0 || i + 1 == num_merges) {
            std::cerr << "PROGRESS " << i + 1 << '\n';
        }
    } 

    assert(vocab.size() <= UINT32_MAX);

    return merges;
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    size_t n;
    uint32_t num_merges;

    if (!(std::cin >> n >> num_merges)) {
        return 1;
    }

    std::vector<std::vector<uint32_t>> words(n);
    std::vector<uint64_t> freqs(n);

    for (size_t i = 0; i < n; ++i) {
        size_t length;

        if (!(std::cin >> length))
            return 1;

        words[i].resize(length);

        for (auto& token : words[i])
            if (!(std::cin >> token))
                return 1;

        if (!(std::cin >> freqs[i]))
            return 1;
    }

    //auto start = std::chrono::steady_clock::now();
    auto merges = train_bpe(words, freqs, num_merges);
    //auto end = std::chrono::steady_clock::now();


    // double seconds = std::chrono::duration<double>(end - start).count();

    //std::cerr << seconds << '\n';

    constexpr char hex[] = "0123456789abcdef";

    auto print_hex = [&](const std::string& s) {
        for (unsigned char c : s) {
            std::cout << hex[c >> 4] << hex[c & 15];
        }
    };

    for (const auto& [a, b] : merges) {
        print_hex(a);
        std::cout << ' ';
        print_hex(b);
        std::cout << '\n';
    }

    return 0;
}