const {defineConfig}=require('eslint/config');
const expo=require('eslint-config-expo/flat');
module.exports=defineConfig([expo,{ignores:['dist/**','dist-android/**']},{rules:{'react-hooks/exhaustive-deps':'off','@typescript-eslint/no-unused-vars':'off'}}]);
