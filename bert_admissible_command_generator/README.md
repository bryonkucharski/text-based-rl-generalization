# Goal
Goal of this model is to extract relevant commands from a text description of a TextWorld state. These commands are encoded with a GRU and ranked in a reinforcement learning framework to maximize reward.

Example Input |
---   | 
DESCRIPTION:  You are in a supermarket. An usual kind of place.  You scan the room for a showcase, and you find a showcase. Why don't you take a picture of it, it'll last longer! The showcase is metallic. On the showcase you see a raw red potato and a black pepper.  There is an open commercial glass door leading south.. INVENTORY: You are carrying: a raw yellow potato a yellow bell pepper

Example Output |
---   |
`close commercial glass door`
`drop yellow bell pepper`
`drop yellow potato`
`eat yellow bell pepper`
`go south`
`put yellow bell pepper on showcase`
`put yellow potato on showcase`
`take black pepper from showcase`
`take red potato from showcase`
